from pylab import *
#from scipy import * # Accessing the leastsq function this way does not work????
from scipy.optimize import leastsq
import numpy


def power_law_curve_fit(xdata, ydata, plot_results=False):

    error_return_value = (None, None)

    if not xdata or not ydata or len(xdata) < 2:
        return error_return_value

    # Define function for calculating a power law
    powerlaw = lambda x, amp, index: amp * (x**index)

    ##########
    # Fitting the data -- Least Squares Method
    ##########

    # Power-law fitting is best done by first converting
    # to a linear equation and then fitting to a straight line.
    #
    #  y = a * x^b
    #  log(y) = log(a) + b*log(x)
    #
    
    logx = log10(xdata)
    logy = log10(ydata)
    
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))

    pinit = [1.0, -1.0]

    out = leastsq(errfunc, pinit, args=(logx, logy), full_output=1)
    if not out[4] in [1, 2, 3, 4] or out[1] == None:
        return error_return_value

    pfinal = out[0]
    covar = out[1]
    #print pfinal
    #print covar

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = sqrt(covar[0][0])
    ampErr = sqrt(covar[1][1]) * amp
    
    ##########
    # Plotting data
    ##########
    if plot_results:
        clf()
        h = subplot(2, 1, 1)
        plot(xdata, ydata)     # Fit
        plot(xdata, powerlaw(xdata, amp, index))     # Fit

        amp_ref = 232.86379468058865
        index_ref = -0.8669993348069398
        xdata = linspace(1.0, 30.1, 30)
        ydata = powerlaw(xdata, amp_ref, index_ref)
        plot(xdata, ydata)     # Fit

        h.set_xlim([0, 30])
        h.set_ylim([0, 260])

        '''
        errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
        text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
        text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
        title('Best Fit Power Law')
        xlabel('X')
        ylabel('Y')
        xlim(1, 11)

        subplot(2, 1, 2)
        loglog(xdata, powerlaw(xdata, amp, index))
        #errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
        xlim(1.0, 11)
        '''
        
        #savefig('power_law_fit.png')
        raw_input("Press Enter to continue...")
        
    return (amp, index)

def get_slope(xdata, ydata1, ydata2, plot_results=False):
    if plot_results:
        clf()
        subplot(2, 1, 1)
        plot(xdata, ydata1)
        subplot(2, 1, 2)
        plot(xdata, ydata2) 
        raw_input("Press Enter to continue...")
    return 0
        

if __name__ == '__main__':
    ##########
    # source from : http://wiki.scipy.org/Cookbook/FittingData
    ##########
    powerlaw = lambda x, amp, index: amp * (x**index)

    ##########
    # Generate data points with noise
    ##########
    num_points = 20

    # Note: all positive, non-zero data
    xdata = linspace(1.1, 10.1, num_points)
    ydata = powerlaw(xdata, 10.0, -2.0)     # simulated perfect data
    yerr = 0.2 * ydata                      # simulated errors (10%)

    ydata += randn(num_points) * yerr       # simulated noisy data
    
    #sys.modules[__name__].__dict__.clear()
    
    amp, index = power_law_curve_fit(xdata, ydata)
    print 'amp : %1.1f index : %1.1f' % (amp, index)
    
    raw_input("Press Enter to continue...")