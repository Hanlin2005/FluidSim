    state0 = CFDState(
        t=jnp.array(0.0),
        w_hat=w0_hat,
        N_hat_prev=jnp.zeros_like(w0_hat),  # no None
        is_first=jnp.array(True)
    )