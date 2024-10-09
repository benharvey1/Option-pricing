import math
import numpy as np
from scipy.stats import norm

def Black_Scholes_option_price(option_type, S, K, r, t, v, q):
    """Function that computes the fair price of a European style option
    using the Black-Scholes model.
    
    Input parameters
    ----------------------------------------------------------------------
    option_type: str
                String representation of the option type. "p" is a put
                option and "c" is a call option.             
    S: float
        Price of underlying assest at current time   
    K: float
        Strike price      
    r:  float
        Risk free interest rate    
    t: float
        Time to expiration date    
    v: float
        Volatility
    q: float
        Constant dividend yield
    ------------------------------------------------------------------------
    Returns: float
            Fair price of option at current time
    """
    # Case where t = 0 (option is expiring)
    if t == 0:
        if option_type == "c":
            return max(S-K, 0)
    
        elif option_type == "p":
            return max(K-S, 0)

    # Compute d_1 and d_2
    d_1 = (math.log(S/K) + (r - q + (v**2)/2)*t)/(v*math.sqrt(t))
    d_2 = d_1 - v*math.sqrt(t)

    # Return option price depending on type
    if option_type == "c":
        return S*norm.cdf(d_1)*math.exp(-q*t) - K*math.exp(-r*t)*norm.cdf(d_2)

    elif option_type == "p":
        return K*math.exp(-r*t)*norm.cdf(-d_2) - S*norm.cdf(-d_1)*math.exp(-q*t)
    
def Compute_Greeks(option_type, S, K, r, t, v, q):
    """Function that computes the Greeks of a European style option
    using the Black-Scholes model.
    
    Input parameters
    ----------------------------------------------------------------------
    option_type: str
                String representation of the option type. "p" is a put
                option and "c" is a call option.           
    S: float
        Price of underlying assest at current time   
    K: float
        Strike price  
    r:  float
        Risk free interest rate   
    t: float
        Time to expiration date  
    v: float
        Volatility
    q: float
        Constant dividend yield

    Outputs
    ------------------------------------------------------------------------
    Delta: float
        Sensitivity of option price to underlying asset price
    Gamma: float
        Sensitivity of Delta to asset price
    Theta: float
        Sensitivity of option price to time
    Vega: float
        Sensitivity of option price to volatility 
    """
    if t != 0:
        # Compute d_1 and d_2
        d_1 = (math.log(S/K) + (r - q + ((v)**2)/2)*t)/(v*math.sqrt(t))
        d_2 = d_1 - v*math.sqrt(t)

        # Return Greeks depending on option type
        if option_type == "c":
            delta = norm.cdf(d_1)*math.exp(-q*t)
            gamma = norm.pdf(d_1)*math.exp(-q*t)/(S*v*math.sqrt(t))
            theta = q*S*math.exp(-q*t)*norm.cdf(d_1) - (S*v*norm.pdf(d_1))/(2*math.sqrt(t)) - r*K*math.exp(-r*t)*norm.cdf(d_2) 
            vega = S*math.sqrt(t)*norm.pdf(d_1)*math.exp(-q*t)

            return delta, gamma, theta, vega

        if option_type == "p":
            delta = -(1-norm.cdf(d_1))*math.exp(-q*t)
            gamma = norm.pdf(d_1)*math.exp(-q*t)/(S*v*math.sqrt(t))
            theta = -(S*v*norm.pdf(d_1))/(2*math.sqrt(t)) + r*K*math.exp(-r*t)*norm.cdf(-d_2) - q*S*math.exp(-q*t)*norm.cdf(-d_1)
            vega = S*math.sqrt(t)*norm.pdf(d_1)*math.exp(-q*t)

            return delta, gamma, theta, vega
    
    else:
        return 0,0,0,0
    
def realised_volatility(price_list, delta_t):
    """Function that computes the standard estimator of realised
    volatility
    
    Input parameters
    ------------------------------------------------------------
    price_list: list
        List of prices of underlying asset at various time steps
    time_steps: list
        List of time steps which asset prices are taken at 

    Output
    ------------------------------------------------------------
    volatility_estimator: float
        standard estimator of volatility of asset price
    """
    N = len(price_list) - 1
    T = N*delta_t
    log_returns = np.zeros(N)
    for i in range(N):
        log_returns[i] = math.log(price_list[i+1]/price_list[i])
    
    mean = np.mean(log_returns)
    variance = sum((r - mean)**2 for r in log_returns)/T

    return math.sqrt(variance)

def implied_volatility(option_type, S, K, r, t, q, market_price, initial_guess=0.2, precision=0.0001, max_steps=1000):
    """Function that computes the implied volatility of an asset
    price using the Black Scholes model and Newton-Raphson method.

    Input parameters
    ----------------------------------------------------------------------
    option_type: str
                String representation of the option type. "p" is a put
                option and "c" is a call option.             
    S: float
        Price of underlying asset at current time   
    K: float
        Strike price      
    r:  float
        Risk-free interest rate    
    t: float
        Time to expiration date    
    q: float
        Constant dividend yield
    market_price: float
        Actual price of the option at current time
    initial_guess: float
        Initial guess for the volatility
    precision: float
        Minimum precision required before stopping
    max_steps: int
        Maximum number of steps before stopping

    Output
    ------------------------------------------------------------
    implied_volatility: float
        Volatility of asset price assuming Black Scholes model
    """
    # Check for valid inputs
    if t <= 0:
        raise ValueError("Time to expiration must be positive.")
    if S <= 0 or K <= 0:
        raise ValueError("Asset price and strike price must be positive.")
    if market_price < 0:
        raise ValueError("Market price cannot be negative.")

    # Initial guess for volatility
    v = initial_guess
    counter = 0

    # Initial price difference
    BS_price = Black_Scholes_option_price(option_type, S, K, r, t, v, q)
    diff = BS_price - market_price

    while abs(diff) > precision and counter < max_steps:
        # Compute Vega
        vega = Compute_Greeks(option_type, S, K, r, t, v, q)[3]

        # Handle small vega to prevent division by zero
        if abs(vega) < 1e-6:
            print(f"Warning: Vega is too small, vega = {vega}. Returning NaN.")
            return np.nan  # Option is likely deep in the money or out of the money

        # Update volatility using Newton-Raphson method
        v -= diff / vega
        counter += 1

        # Recalculate the option price and the difference
        BS_price = Black_Scholes_option_price(option_type, S, K, r, t, v, q)
        diff = BS_price - market_price

    # If it doesn't converge, return NaN
    if counter == max_steps:
        print("Warning: Maximum steps reached without convergence.")
        return np.nan

    return v

def Monte_Carlo_option_price(option_type, S, K, r, t, v, q, num_simulations=10000):
    """Function that computes the fair price of a European style option
    using Monte Carlo simulations.
    
    Input parameters
    ----------------------------------------------------------------------
    option_type: str
                String representation of the option type. "p" is a put
                option and "c" is a call option.             
    S: float
        Price of underlying assest at current time   
    K: float
        Strike price      
    r:  float
        Risk free interest rate    
    t: float
        Time to expiration date    
    v: float
        Volatility
    q: float
        Constant dividend yield
    num_simulations: int
        number of monte carlo simulations to carry out
    num_time_steps: int
        number of time steps in each simulation
    ------------------------------------------------------------------------
    Returns: float
            Fair price of option at current time
    """
    dt = 1/365.25
    num_time_steps = int(t/dt)
    payoffs = []

    for _ in range(num_simulations):
        S_T = S
        for _ in range(num_time_steps):
            z = np.random.normal()
            S_T *= math.exp((r - q - 0.5*v**2)*dt + v*math.sqrt(dt)*z)

        if option_type == 'c':
            payoffs.append(max(S_T - K, 0))

        elif option_type == 'p':
            payoffs.append(max(K - S_T, 0))

    # Under risk neutral measure current value of asset is just expected value
    # of future payoff taken wrt that same risk neutral measure

    price = math.exp(-r*t)*np.mean(payoffs)

    return price

def Monte_Carlo_stock_paths(S, r, t, v, q, num_simulations=10000):
    """Function that returns an array of price paths for an asset using Monte Carlo simulations.
    
    Input parameters
    ----------------------------------------------------------------------           
    S: float
        Price of assest at current/initial time        
    r:  float
        Risk free interest rate 
    t: float
        time to 'expiration' (final time - initial time)
    v: float
        Volatility
    q: float
        Constant dividend yield
    num_simulations: int
        number of monte carlo simulations to carry out
    num_time_steps: int
        number of time steps in each simulation
    ------------------------------------------------------------------------
    Returns: numpy array
        list of stock prices for each simulation in form of
        numpy array       
    """
    dt = 1/365.25
    num_time_steps = int(t/dt)
    prices = np.zeros((num_simulations, num_time_steps))

    prices[:,0] = S # Initial price

    for n in range(num_simulations):
        for t in range(1, num_time_steps):
            z = np.random.normal()
            prices[n, t] = prices[n,t-1]*math.exp((r - q - 0.5*v**2)*dt + v*math.sqrt(dt)*z)

    return prices










    
