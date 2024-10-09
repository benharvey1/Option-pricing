import math
import cmath
import numpy as np
from scipy import integrate

# dS = r*S*dt + sqrt(v)*S*dW_1
# dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_2

def Heston_stock_paths(S, r, t, v, kappa, theta, xi, rho, num_simulations=10000):
    """Function that returns an array of stock price paths using the Heston model.
    
    Input parameters
    ----------------------------------------------------------------------             
    S: float
        Price of underlying assest at current time         
    r:  float
        Risk free interest rate    
    t: float
        Time to expiration date    
    v: float
        initial volatility
    kappa: float
        mean reversion rate
    theta: float
        long term volatility mean
    xi: float
        volatility of volatility
    rho: float
        Pearson correlation coefficient
    num_simulations: int
        number of Monte Carlo simulations
    num_time_steps: int
        number of time steps for each simulation
        
    Outputs
    ------------------------------------------------------------------------
    prices: numpy array
        list of stock prices for each simulation in form of
        numpy array
    volatilities: numpy array
        list of volatilities for each simulation in form of
        numpy array
    """
    dt = 1/365.25
    num_time_steps = int(t/dt)
    prices = np.zeros((num_simulations, num_time_steps+1))
    volatilities = np.zeros((num_simulations, num_time_steps+1))

    prices[:, 0] = S # Initial price
    volatilities[:, 0] = v  # Initial volatility

    for n in range(num_simulations):
        for t in range(1, num_time_steps+1):

            z_1 = np.random.normal()
            z_2 = rho*z_1 + math.sqrt(1 - rho**2)*np.random.normal()

            volatilities[n, t] = np.abs(volatilities[n, t-1] + kappa*(theta - volatilities[n, t-1])*dt + xi*math.sqrt(volatilities[n, t-1]*dt)*z_2)
            prices[n, t] = prices[n, t-1] + r*prices[n, t-1]*dt + math.sqrt(volatilities[n, t-1]*dt)*prices[n, t-1]*z_1

    return prices, volatilities

def Heston_option_prices_Monte_Carlo(option_type, S, K, r, t, v, kappa, theta, xi, rho, num_simulations=10000):
    """Function that returns fair price of European style option
    (with non-dividend paying asset) using the Heston model.
    
    Input parameters
    ----------------------------------------------------------------------             
    option_type: str
        string representation of option type
    S: float
        Price of underlying assest at current time
    K: float
        Strike price of underlying asset      
    r:  float
        Risk free interest rate    
    t: float
        Time to expiration date    
    v: float
        initial volatility
    kappa: float
        mean reversion rate
    theta: float
        long term volatility mean
    xi: float
        volatility of volatility
    rho: float
        Pearson correlation coefficient
    num_simulations: int
        number of Monte Carlo simulations
    num_time_steps: int
        number of time steps for each simulation
        
    Outputs
    ------------------------------------------------------------------------
    price: float
        Fair price of option at current time   
    """

    prices, volatilities = Heston_stock_paths(S, r, t, v, kappa, theta, xi, rho, num_simulations)
    prices_at_expiration = prices[:, -1]
    payoffs = []
    
    for S_T in prices_at_expiration:

        if option_type == 'c':
            payoffs.append(max(S_T - K, 0))

        elif option_type == 'p':
            payoffs.append(max(K - S_T, 0))

    price = math.exp(-r*t)*np.mean(payoffs)

    return price

def Heston_characteristic_function(S, r, t, phi, v0, kappa, theta, xi, rho, status):
    """Function that computes the characteristic function of the Heston model.
    
     Input parameters
    ----------------------------------------------------------------------             
    S: float
        Price of underlying assest at current time         
    r:  float
        Risk free interest rate    
    t: float
        Time to expiration date
    phi: float

    v0: float
        initial volatility
    kappa: float
        mean reversion rate
    theta: float
        long term volatility mean
    xi: float
        volatility of volatility
    rho: float
        Pearson correlation coefficient
    status: int
        Takes value 1 or 2
    
    Outputs
    ------------------------------------------------------------------------
    f: float
        Value of Characteristc function  """
    a = kappa * theta
    
    if status == 1:
        u = 1/2
        b = kappa - rho*xi
    
    elif status == 2:
        u = -1/2
        b = kappa

    d = cmath.sqrt((1j*rho*xi*phi - b)**2 - (xi**2)*(2*1j*u*phi - phi**2))
    g = (b - 1j*rho*xi*phi + d)/(b - 1j*rho*xi*phi - d)
    C = 1j*r*phi*t + (a/xi**2)*((b - 1j*rho*xi*phi + d)*t - 2*cmath.log((1 - g*cmath.exp(t*d))/(1-g)))
    D = (b - 1j*rho*xi*phi + d)/(xi**2) * (1 - cmath.exp(d*t))/(1 - g*cmath.exp(d*t))

    return cmath.exp(C + v0*D + 1j*phi*cmath.log(S))

def Pi(S, K, r, t, v0, kappa, theta, xi, rho, status):
    """Function that computes the Pi functions of the Heston model.
    
     Input parameters
    ----------------------------------------------------------------------             
    S: float
        Price of underlying assest at current time  
    K: float
        Strike price       
    r:  float
        Risk free interest rate    
    t: float
        Time to expiration date
    phi: float

    v0: float
        initial volatility
    kappa: float
        mean reversion rate
    theta: float
        long term volatility mean
    xi: float
        volatility of volatility
    rho: float
        Pearson correlation coefficient
    status: int
        Takes value 1 or 2
    
    Outputs
    ------------------------------------------------------------------------
    Pi: float
        Value of Pi function  """
    def integrand(phi):
        complex_integrand = (cmath.exp(-1j*phi*cmath.log(K))*Heston_characteristic_function(S, r, t, phi, v0, kappa, theta, xi, rho, status)/(1j*phi))
        return complex_integrand.real
    
    result, _ = integrate.quad(integrand, 0, 100)
    Pi = 0.5 + result / np.pi
    return Pi

def Heston_option_price(option_type, S, K, r, t, v0, kappa, theta, xi, rho):
    """Function that returns fair price of European style option
    (with non-dividend paying asset) using the Heston model.
    
    Input parameters
    ----------------------------------------------------------------------             
    option_type: str
        string representation of option type
    S: float
        Price of underlying assest at current time
    K: float
        Strike price of underlying asset      
    r:  float
        Risk free interest rate    
    t: float
        Time to expiration date    
    v0: float
        initial volatility
    kappa: float
        mean reversion rate
    theta: float
        long term volatility mean
    xi: float
        volatility of volatility
    rho: float
        Pearson correlation coefficient
        
    Outputs
    ------------------------------------------------------------------------
    price: float
        Fair price of option at current time   
    """
    P1 = Pi(S, K, r, t, v0, kappa, theta, xi, rho, 1)
    P2 = Pi(S, K, r, t, v0, kappa, theta, xi, rho, 2)

    if option_type == 'c':
        return S*P1 - K*P2*math.exp(-r*t)
    
    if option_type == 'p':
        return K*math.exp(-r*t)*(1-P2) + S*(P1-1)