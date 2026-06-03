import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

@tf.custom_gradient
def log_pdf_dalpha(y, alpha, c, beta):
    eps = 1.0e-7
    log_y = tf.math.log(y + eps)
    
    # First derivative w.r.t alpha: log(beta) - digamma(alpha) + c * log(y)
    first_derivative_alpha = tf.math.log(beta + eps) - tf.math.digamma(alpha) + c * log_y

    def custom_second_derivative(upstream_grad):
        # Second derivatives
        d2log_pdf_y_alpha2 = -tf.math.polygamma(1.0, alpha)
        d2log_pdf_y_alphac = log_y
        d2log_pdf_y_alphabeta = 1.0 / (beta + eps)

        jac_alpha2 = upstream_grad * d2log_pdf_y_alpha2
        jac_alphac = upstream_grad * d2log_pdf_y_alphac
        jac_alphabeta = upstream_grad * d2log_pdf_y_alphabeta
        
        # Returns derivatives for: y, alpha, c, beta
        return None, jac_alpha2, jac_alphac, jac_alphabeta
    
    return first_derivative_alpha, custom_second_derivative

@tf.custom_gradient
def log_pdf_dc(y, alpha, c, beta):
    eps = 1.0e-7
    y_safe = y + eps
    log_y = tf.math.log(y_safe)
    y_pow_c = tf.math.pow(y_safe, c)
    
    # First derivative w.r.t c: 1/c + alpha*log(y) - beta * y^c * log(y)
    first_derivative_c = 1.0 / (c + eps) + alpha * log_y - beta * y_pow_c * log_y

    def custom_second_derivative(upstream_grad):
        d2log_pdf_y_c2 = -1.0 / (c**2.0 + eps) - beta * y_pow_c * (log_y**2.0)
        d2log_pdf_y_alphac = log_y
        d2log_pdf_y_cbeta = -y_pow_c * log_y

        jac_c2 = upstream_grad * d2log_pdf_y_c2
        jac_alphac = upstream_grad * d2log_pdf_y_alphac
        jac_cbeta = upstream_grad * d2log_pdf_y_cbeta
        
        # Returns derivatives for: y, alpha, c, beta
        return None, jac_alphac, jac_c2, jac_cbeta
    
    return first_derivative_c, custom_second_derivative


@tf.custom_gradient
def log_pdf_dbeta(y, alpha, c, beta):
    eps = 1.0e-7
    y_pow_c = tf.math.pow(y + eps, c)
    
    # First derivative w.r.t beta: alpha/beta - y^c
    first_derivative_beta = alpha / (beta + eps) - y_pow_c

    def custom_second_derivative(upstream_grad):
        d2log_pdf_y_beta2 = -alpha / (beta**2.0 + eps)
        d2log_pdf_y_alphabeta = 1.0 / (beta + eps)
        d2log_pdf_y_cbeta = -y_pow_c * tf.math.log(y + eps)

        jac_beta2 = upstream_grad * d2log_pdf_y_beta2
        jac_alphabeta = upstream_grad * d2log_pdf_y_alphabeta
        jac_cbeta = upstream_grad * d2log_pdf_y_cbeta
        
        # Returns derivatives for: y, alpha, c, beta
        return None, jac_alphabeta, jac_cbeta, jac_beta2
    
    return first_derivative_beta, custom_second_derivative

@tf.custom_gradient
def log_pdf(y, alpha, c, beta):
    # Enforce float32
    y = tf.cast(y, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    c = tf.cast(c, tf.float32)
    beta = tf.cast(beta, tf.float32)

    eps = 1.0e-7
    y_safe = y + eps
    log_y = tf.math.log(y_safe)
    
    # Log-Likelihood value based on: f(y) = [c * beta^alpha / Gamma(alpha)] * y^(c*alpha - 1) * exp(-beta * y^c)
    term1 = tf.math.log(c + eps) + alpha * tf.math.log(beta + eps) - tf.math.lgamma(alpha)
    term2 = (c * alpha - 1.0) * log_y
    term3 = -beta * tf.math.pow(y_safe, c)
    
    log_pdf_y = term1 + term2 + term3

    def custom_derivative(upstream_grad):
        dlog_pdf_y_alpha = log_pdf_dalpha(y_safe, alpha, c, beta)
        dlog_pdf_y_c = log_pdf_dc(y_safe, alpha, c, beta)
        dlog_pdf_y_beta = log_pdf_dbeta(y_safe, alpha, c, beta)
        
        grad_alpha = upstream_grad * dlog_pdf_y_alpha
        grad_c = upstream_grad * dlog_pdf_y_c
        grad_beta = upstream_grad * dlog_pdf_y_beta
        
        # Returns derivatives for: y, alpha, c, beta
        return None, grad_alpha, grad_c, grad_beta
              
    return log_pdf_y, custom_derivative


@tf.function(reduce_retracing=True)
def pdf(y, alpha, c, beta):
    y = tf.cast(y, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    c = tf.cast(c, tf.float32)
    beta = tf.cast(beta, tf.float32)

    eps = 1.0e-7
    y_safe = y + eps
    
    # Direct exponential formulation for numerical stability
    log_f = log_pdf(y_safe, alpha, c, beta)
    return tf.math.exp(log_f)

@tf.function(reduce_retracing=True)
def cdf(y, alpha, c, beta):
    y = tf.cast(y, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    c = tf.cast(c, tf.float32)
    beta = tf.cast(beta, tf.float32)

    eps = 1.0e-7
    y_safe = y + eps
    
    # T ~ Gamma(alpha, beta), y^c = t
    dist = tfp.distributions.Gamma(concentration = alpha, rate = beta)
    return dist.cdf(tf.math.pow(y_safe, c))

@tf.function(reduce_retracing=True)
def ppf(q, alpha, c, beta):
    q = tf.cast(q, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    c = tf.cast(c, tf.float32)
    beta = tf.cast(beta, tf.float32)
    
    # Transform the quantile of the Gamma distribution back to the GG space
    dist = tfp.distributions.Gamma(concentration = alpha, rate = beta)
    t_q = dist.quantile(q)
    
    # y = t^(1/c)
    y_q = tf.math.pow(t_q, 1.0 / c)
    return y_q

@tf.function(reduce_retracing=True)
def S(y, alpha, c, beta):
    y = tf.cast(y, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    c = tf.cast(c, tf.float32)
    beta = tf.cast(beta, tf.float32)

    eps = 1.0e-7
    y_safe = y + eps
    
    # T ~ Gamma(alpha, beta), y^c = t
    dist = tfp.distributions.Gamma(concentration = alpha, rate = beta)
    
    # Native survival function is highly stable for the extreme right tail
    return dist.survival_function(tf.math.pow(y_safe, c))

@tf.function(reduce_retracing=True)
def log_S(y, alpha, c, beta):
    y = tf.cast(y, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    c = tf.cast(c, tf.float32)
    beta = tf.cast(beta, tf.float32)

    eps = 1.0e-7
    y_safe = y + eps
    
    dist = tfp.distributions.Gamma(concentration = alpha, rate = beta)
    
    # Safely computes the log of the survival probability without underflowing to -inf
    return dist.log_survival_function(tf.math.pow(y_safe, c))