### Gaussian process-based disease progression modeling and time-shift estimation.
#
#  - The software iteratively estimates monotonic progressions for each biomarker and realigns the individual observations in time
#   Basic usage:
#
#       model = GP_progression_model.GP_progression_model(input_X,input_N,N_random_features)
#
#   X and Y should be A list of biomarkers arrays. Each entry "i" of the list is a list of individuals' observations for the biomarker i
#   The monotonicity is enforced by the parameter self.penalty (higher -> more monotonic)
#
# - The class comes with an external method for transforming a given .csv file in the required input X and Y:
#
#       X,Y,list_biomarker = GP_progression_model.convert_csv(file_path)
#
# - The method Save(folder_path) saves the model parameters to an external folder, that can be subsequently read with the
# method Load(folder_path)
#
# - Optimization can be done with the method Optimize:
#
#       model.Optimize()
#
# This software is based on the publication:
#
# Disease Progression Modeling and Prediction through Random Effect Gaussian Processes and Time Transformation
# Marco Lorenzi, Maurizio Filippone, Daniel C. Alexander, Sebastien Ourselin
# arXiv:1701.01668
#
# Gaussian process regression based on random features approximations is based on the paper:
#
# Random Feature Expansions for Deep Gaussian Processes (ICML 2017, Sydney)
# K. Cutajar, E. V. Bonilla, P. Michiardi, M. Filippone
# arXiv:1610.04386


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

class GP_progression_model(object):
    plt.interactive(False)
    def __init__(self, X,Y,  N_rnd_features=10, names_biomarkers = [], group = []):

        #Initializing variables

        self.names_biomarkers = names_biomarkers
        self.group = group
        self.N_rnd_features = int(N_rnd_features)
        self.N_samples = len(X[0])
        self.N_biom = len(X)
        self.X_array = []
        self.Y_array = []
        self.mean_std_X = []
        self.mean_std_Y = []
        self.max_X = []
        self.max_Y = []
        self.N_obs_per_sub = []
        self.params_time_shift = np.ndarray([2,len(X[0])])

        # Time shift initialized to 0
        self.params_time_shift[0, :] = 0

        # Estension of the model will include a time scaling factor (fixed to 1 so far)
        self.params_time_shift[1,:] = 1

        for l in range(self.N_biom):
            # Creating 1d arrays of individuals' time points and observations
            self.X_array.append([np.float(item) for sublist in X[l] for item in sublist])
            self.Y_array.append([np.float(item) for sublist in Y[l] for item in sublist])
            self.N_obs_per_sub.append([len(X[l][j]) for j in range(len(X[l]))])

        self.rescale()
        self.minX = np.float(np.min([el for sublist in self.X_array for item in sublist for el in item]))
        self.maxX = np.float(np.max([el for sublist in self.X_array for item in sublist for el in item]))

        # Number of derivative points uniformely distributed on the X axis
        self.N_Dpoints = 20
        self.DX = np.linspace(self.minX,self.maxX,self.N_Dpoints).reshape([self.N_Dpoints,1])

        # Initializing random features for kernel approximation
        self.perturbation_Omega = np.random.randn(self.N_rnd_features)

        self.init_params_var = []
        self.init_params_full = []

        # Monotonicity constraint (higher -> more monotonic)
        self.penalty = []

        # Initializing fixed effect parameters per biomarkers to default values
        for l in range(self.N_biom):
            self.init_params_var.append(np.concatenate([ np.zeros([ self.N_rnd_features]) - 1, np.zeros([ self.N_rnd_features]) - 1, np.zeros([ 2 * self.N_rnd_features]) , np.zeros([ 2 * self.N_rnd_features])]))
            sigma = 0
            length_scale = -3
            eps = -4
            self.init_params_full.append(np.concatenate([self.init_params_var[l], np.array([sigma]), np.array([length_scale]), np.array([eps])]))
            self.penalty.append(1)

        self.parameters = []
        for l in range(self.N_biom):
            self.parameters.append(self.init_params_full[l])


        # Initializing individuals random effects
        self.rand_parameters = []
        self.rand_parameter_type = []

        for biom in range(self.N_biom):
            self.rand_parameter_type.append([])
            self.rand_parameters.append([])
            for sub in range(self.N_samples):
                if self.N_obs_per_sub[biom][sub]==0:
                    self.rand_parameter_type[biom].append(0)
                    self.rand_parameters[biom].append(0)
                elif self.N_obs_per_sub[biom][sub] < 3:
                    self.rand_parameter_type[biom].append(1)
                    self.rand_parameters[biom].append(0)
                else:
                    self.rand_parameter_type[biom].append(2)
                    self.rand_parameters[biom].append([0,0])

                obs = np.array([self.X_array[biom][k][0] for k in range(int(np.sum(self.N_obs_per_sub[biom][:sub])),
                                                               np.sum(self.N_obs_per_sub[biom][:sub + 1]))])



    def rescale(self):
        # Standardizes X and Y axes and saves the rescaling parameters for future output
        for l in range(self.N_biom):
            self.X_array[l] = np.array(self.X_array[l]).reshape([len(self.X_array[l]),1])
            self.Y_array[l] = np.array(self.Y_array[l]).reshape([len(self.Y_array[l]), 1])
            sd = np.std(self.X_array[l])
            if (sd > 0):
                self.mean_std_X.append([np.mean(self.X_array[l]), np.std(self.X_array[l])])
            else:
                self.mean_std_X.append([np.mean(self.X_array[l]), 1])

            self.mean_std_Y.append([np.mean(self.Y_array[l]), np.std(self.Y_array[l])])
            self.X_array[l] = (self.X_array[l] - self.mean_std_X[l][0])/ self.mean_std_X[l][1]
            if np.max(self.X_array[l]) > 0:
                self.max_X.append(np.max(self.X_array[l]))
            else:
                self.max_X.append(1)
            self.X_array[l] = self.X_array[l]/self.max_X[l]
            self.Y_array[l] = (self.Y_array[l] - self.mean_std_Y[l][0]) / self.mean_std_Y[l][1]
            self.max_Y.append(np.max(self.Y_array[l]))
            self.Y_array[l] = self.Y_array[l] / self.max_Y[l]

    def Set_penalty(self, penalty):
        for l in range(self.N_biom):
            self.penalty[l] = penalty

    def Reset_parameters(self):
        # Reset paramters to standard values
        self.init_params_var = []
        self.init_params_full = []
        for l in range(self.N_biom):
            self.init_params_var.append(np.concatenate([ np.zeros([ self.N_rnd_features]) - 1, np.zeros([ self.N_rnd_features]) - 1, np.zeros([ 2 * self.N_rnd_features]) , np.zeros([ 2 * self.N_rnd_features])]))
            sigma = -1

            if (self.maxX==self.minX):
                length_scale = 0
            else:
                length_scale = np.log((self.maxX - self.minX)/8)

            eps = -3
            self.init_params_full.append(np.concatenate([self.init_params_var[l], np.array([sigma]), np.array([length_scale]), np.array([eps])]))

        self.parameters = []
        for l in range(self.N_biom):
            self.parameters.append(self.init_params_full[l])

    def phi(self, X, omega, sigma):
        # Random feature expansion in cosine a sine basis
        return np.sqrt(sigma) / np.sqrt(len(omega)) * np.concatenate([np.cos(omega * X), np.sin(omega * X)], axis=1)

    def Dphi(self, X, omega, sigma):
        # Derivative of the random feature expansion with respect to X
        return np.sqrt(sigma) / np.sqrt(len(omega)) * np.concatenate([- np.sin(omega * X) * omega, np.cos(omega * X) * omega], axis=1)

    def DDphi_omega(self, X, omega, sigma):
        # Double derivative of the random feature expansion with respect to X and omega
        return np.sqrt(sigma) / np.sqrt(len(omega)) * \
               np.concatenate([- np.cos(omega * X) * omega * X - np.sin(omega * X)  ,\
                               - np.sin(omega * X) * omega * X + np.cos(omega * X) ], axis=1)

    def Dphi_omega(self, X, omega, sigma):
        # Derivative of the random feature expansion with respect to omega
        return np.sqrt(sigma) / np.sqrt(len(omega)) * np.concatenate([- np.sin(omega * X) * X , np.cos(omega * X) * X ], axis=1)

    def Dphi_time_shift(self, X, omega, sigma):
        # Derivative of the random feature expansion with respect to time-shift parameters
        return np.sqrt(sigma) / np.sqrt(len(omega)) * np.concatenate([- np.sin(omega * X) , np.cos(omega * X) ], axis=1)

    def basis(self, X, sigma, random_weights):
        return self.phi(X, random_weights, sigma)

    def Dbasis_omega(self, X, sigma, random_weights):
        return self.Dphi_omega(X, random_weights, sigma)

    def Dbasis(self, X, sigma, random_weights):
        return self.Dphi(X, random_weights, sigma)

    def DDbasis_omega(self, X, sigma, random_weights):
        return self.DDphi_omega(X, random_weights, sigma)

    def Dbasis_time_shift(self, X, sigma, random_weights):
        return self.Dphi_time_shift(X, random_weights, sigma)

    def KL(self, s_omega, m_omega,  s, m, l):
        # Kullback Leibler divergence for random features and weights distributions
        termOmega = np.sum(0.5 * (np.log(1 / (l * s_omega)) - 1 + (l * s_omega) + l * m_omega ** 2))
        termW = np.sum(0.5 * (np.log(1 / s) - 1 + s + m ** 2))
        return termW + termOmega

    def unpack_parameters(self, params):
        s_omega = params[:self.N_rnd_features].reshape([self.N_rnd_features,1])
        m_omega = params[self.N_rnd_features:2*self.N_rnd_features].reshape([self.N_rnd_features,1])
        s_w = params[2*self.N_rnd_features:4*self.N_rnd_features].reshape([2*self.N_rnd_features,1])
        m_w = params[4*self.N_rnd_features:6*self.N_rnd_features].reshape([2*self.N_rnd_features,1])
        sigma = params[6*self.N_rnd_features]
        length_scale = params[6*self.N_rnd_features+1]
        eps = params[6*self.N_rnd_features+2]
        return s_omega, m_omega, s_w, m_w, sigma, length_scale, eps

    def log_posterior(self, X,Y, N, perturbationW, params, penalty):
        # NOT USED
        # Returns log-posterior for a given set of biomarker's paramters and a random perturbation of the weights W
        s_omega, m_omega, s_w, m_w, sigma, l, eps = self.unpack_parameters(params)
        s_omega = np.exp(s_omega)
        s_w = np.exp(s_w)
        l = np.exp(l)
        sigma = np.exp(sigma)
        eps= np.exp(eps)

        W = np.multiply(perturbationW, np.sqrt(s_w)) + m_w
        Omega = 1/np.sqrt(l) * self.perturbation_Omega
        output = self.basis(X, sigma, Omega)
        Doutput = self.Dbasis(self.DX, sigma, Omega)
        Kullback_Leibler = self.KL( s_omega, m_omega, s_w, m_w, l)

        Dterm = np.sum(penalty * np.dot(Doutput, W) - np.log(1 + np.exp(penalty * np.dot(Doutput, W))))

        prior = (eps - 0.3) ** 2 / 1e-2 + (sigma - 0.5) ** 2 / 1e-2

        return -0.5 *  ( np.log(2 * np.pi * eps) + np.sum((Y - np.dot(output,W))**2)/eps) - Kullback_Leibler  - prior + Dterm

    def log_posterior_grad(self, X,Y, N, perturbationW, params, penalty):
        # Input: X, Y and a biomarker's parameters, random perturbation of the weights W
        # Output: log-posterior and parameters gradient
        s_omega, m_omega, s_w, m_w, sigma, l, eps = self.unpack_parameters(params)
        s_omega = np.exp(s_omega)
        s_w = np.exp(s_w)
        l = np.exp(l)
        sigma = np.exp(sigma)
        eps= np.exp(eps)

        W = np.multiply(perturbationW, np.sqrt(s_w)) + m_w
        Omega = 1/np.sqrt(l) * self.perturbation_Omega
        output = self.basis(X, sigma, Omega)
        Doutput = self.Dbasis(self.DX, sigma, Omega)
        Kullback_Leibler = self.KL( s_omega, m_omega, s_w, m_w, l)

        # Modify the prior length scale according to current X range
        prior_length_scale = (self.maxX-self.minX)/3

        Dterm = np.sum(penalty * np.dot(Doutput, W) - np.log(1 + np.exp(penalty * np.dot(Doutput, W))))
        prior = (eps - 1) ** 2 / 1e-2 + (sigma - 2) ** 2 / 1e-3  + (l - prior_length_scale)**2/1e-2

        posterior = -0.5 *  ( np.log(2 * np.pi * eps) + np.sum((Y - np.dot(output,W))**2)/eps) - Kullback_Leibler  - prior + Dterm

        # Derivative of weights mean ad sd
        d_m_w = np.dot(((Y - np.dot(output,W))).T,output)/eps + penalty * np.sum(Doutput,0) \
                - np.multiply(1/(1 + np.exp(penalty * np.dot(Doutput, W))), np.exp(penalty * np.dot(Doutput, W))*penalty * np.dot(Doutput, W)).T.dot(Doutput)\
                -  m_w.T

        d_s_w = np.multiply(np.dot(((Y - np.dot(output,W))).T,output),0.5*np.multiply(perturbationW, np.sqrt(s_w)).T)/eps \
                + np.multiply(penalty * np.sum(Doutput,0), 0.5*np.multiply(perturbationW, np.sqrt(s_w)).T)\
                - np.multiply(0.5*np.multiply(perturbationW, np.sqrt(s_w)).T, \
                np.multiply(1/(1 + np.exp(penalty * np.dot(Doutput, W))), np.exp(penalty * np.dot(Doutput, W))*penalty * np.dot(Doutput, W)).T.dot(Doutput))\
                + 0.5 * (-1 + s_w).T


        Doutput_omega = self.Dbasis_omega(X, sigma, Omega)
        DDoutput_omega = self.DDbasis_omega(self.DX, sigma, Omega)

        grad_prod = - 0.5 * np.dot(np.multiply(Doutput_omega, np.tile(self.perturbation_Omega, 2) * 1 / np.sqrt(l)), W)
        grad2_prod = - 0.5 * np.dot(np.multiply(DDoutput_omega, np.tile(self.perturbation_Omega, 2) * 1 / np.sqrt(l)), W)

        # Derivative of length scale
        d_l = - 2* np.sum( np.multiply(((Y - np.dot(output,W)))/eps, grad_prod)) \
              + penalty  *  np.sum( grad2_prod ) \
              - np.sum(np.multiply(np.multiply(np.exp(penalty * np.dot(Doutput, W)), 1 / (1 + np.exp(penalty * np.dot(Doutput, W)))), \
            penalty * grad2_prod))\
            -  2* (l - prior_length_scale)/1e-2 * l

        # Derivative of amplitude
        d_sigma = + np.sum(np.multiply(((Y - np.dot(output,W))).T/eps,np.dot(output,W).T/np.sqrt(sigma))) * np.sqrt(sigma) \
                  -  0.5* penalty  *  np.sum(np.dot(Doutput, W)) \
                  + np.sum(np.multiply( np.multiply(np.exp(penalty * np.dot(Doutput, W)), 1 / (1 + np.exp(penalty * np.dot(Doutput, W)))), \
                                        0.5 * penalty * (np.dot(Doutput, W))))\
                  - 2* (sigma - 2)/1e-2 * sigma

        # Derivative of noise term
        d_eps = + 0.5 *  ( 1 + np.sum((Y - np.dot(output,W))**2)/eps) - 2* (eps - 0.5) / 1e-2 * eps

        # Derivative of penalization parameter
        d_penalty = np.sum(np.dot(Doutput, W)) \
                    - np.sum( np.multiply( np.multiply(np.dot(Doutput, W),np.exp(penalty * np.dot(Doutput, W))), \
                                1/(1 + np.exp(penalty * np.dot(Doutput, W)))))


        return posterior, np.hstack([np.repeat(0,len(s_omega)).flatten(), np.repeat(0,len(m_omega)).flatten(), d_s_w.flatten(), d_m_w.flatten(),  np.array([d_sigma]), np.array(d_l), np.array([d_eps])]), d_penalty


    def stochastic_grad_manual(self, params):
        # Stochastic gradient of log-posterior with respect ot given parameters
        # Default number of MC samples is 100
        output_MC_grad = []
        output_loglik = []
        output_grad_penalty = []
        for l in range(self.N_biom):
            current_params = params[l]
            current_X = self.X_array[l]
            current_Y = self.Y_array[l]
            MC_grad = np.zeros(len(params[l]))
            output_grad_penalty.append(0)
            loglik = 0
            for j in range(100):
                perturbation_W = np.random.randn( 2 * self.N_rnd_features).reshape(\
                                                                            [ 2*self.N_rnd_features,1])
                objective_cost_function = lambda params: \
                    self.log_posterior_grad(current_X, current_Y,self.N_rnd_features, perturbation_W, params, self.penalty[l])

                value, grad, grad_penalty = objective_cost_function(current_params)
                MC_grad = MC_grad - grad
                loglik = loglik - value
                output_grad_penalty[l] = output_grad_penalty[l] - grad_penalty
            output_MC_grad.append(MC_grad/100)
            output_loglik.append(loglik/100)
            output_grad_penalty[l] = output_grad_penalty[l]/100
        return output_loglik, output_MC_grad, output_grad_penalty

    def stochastic_grad_manual_mini_batch(self, params, batch_size):
        # Mini-batch implementation of stochastic gradient of log-posterior with respect ot given parameters
        # Default number of MC samples is 100
        sample_batch = np.random.choice(range(self.N_samples), batch_size)
        output_MC_grad = []
        output_loglik = []
        output_grad_penalty = []
        for l in range(self.N_biom):
            Xdata = np.array([[1e10]])
            Ydata = np.array([[1e10]])
            for sub in range(self.N_samples):
                if np.in1d(sub,sample_batch):
                    temp = self.X_array[l][int(np.sum(self.N_obs_per_sub[l][:sub])):np.sum(self.N_obs_per_sub[l][:sub+1])]
                    Xdata = np.hstack([Xdata,temp.T])
                    tempY = self.Y_array[l][int(np.sum(self.N_obs_per_sub[l][:sub])):np.sum(self.N_obs_per_sub[l][:sub + 1])]
                    Ydata = np.hstack([Ydata, tempY.T])

            Xdata = Xdata[0][1:].reshape([len(Xdata[0][1:]), 1])
            Ydata = Ydata[0][1:].reshape([len(Ydata[0][1:]), 1])

            current_params = params[l]
            current_X = Xdata
            current_Y = Ydata
            MC_grad = np.zeros(len(params[l]))
            output_grad_penalty.append(0)
            loglik = 0
            for j in range(100):
                perturbation_W = np.random.randn( 2 * self.N_rnd_features).reshape(\
                                                                            [ 2*self.N_rnd_features,1])
                objective_cost_function = lambda params: \
                    self.log_posterior_grad(current_X, current_Y,self.N_rnd_features, perturbation_W, params, self.penalty[l])

                value, grad, grad_penalty = objective_cost_function(current_params)
                MC_grad = MC_grad - grad
                loglik = loglik - value
                output_grad_penalty[l] = output_grad_penalty[l] - grad_penalty
            output_MC_grad.append(float(len(Xdata)) / len(self.X_array[l]) * MC_grad / 100)
            output_loglik.append(float(len(Xdata)) / len(self.X_array[l]) * loglik / 100)
            output_grad_penalty[l] = float(len(Xdata)) / len(self.X_array[l]) * output_grad_penalty[l] / 100
        return output_loglik, output_MC_grad, output_grad_penalty



    def Adadelta(self, Niterat, objective_grad, learning_rate, init_params, output_grad_penalty = False):
        # Adadelta optimizer
        params = []
        diag = []

        if output_grad_penalty:
            param_penalty = []
            diag_penalty = []

        for l in range(self.N_biom):
            params.append(init_params[l].copy())
            diag.append(np.zeros(len(params[l])))
            if output_grad_penalty:
                param_penalty.append(0)
                diag_penalty.append(0)

        epsilon = 1e-8

        for i in range(Niterat):
            fun_value, fun_grad, fun_grad_penalty = objective_grad(params)
            for l in range(self.N_biom):
                diag[l] = 0.9 * diag[l] + 0.1 * fun_grad[l] ** 2
                params[l] = params[l] - np.multiply(learning_rate * fun_grad[l], 1 / np.sqrt(diag[l] + epsilon))

                if output_grad_penalty:
                    diag_penalty[l] = 0.9 * diag_penalty[l] + 0.1 * fun_grad_penalty[l] ** 2
                    param_penalty[l] = param_penalty[l] - learning_rate * fun_grad_penalty[l]/ np.sqrt(diag_penalty[l] + epsilon)

            print i, fun_value

            for l in range(self.N_biom):
                self.parameters[l] = params[l]

                if output_grad_penalty:
                    self.penalty [l]= param_penalty[l]

    def Optimize_GP_parameters(self, optimize_penalty =False, Niterat = 10):
        # Method for optimization of GP parameters (weights, length scale, amplitude and noise term)
        self.Reset_parameters()
        #objective_grad = lambda params: self.stochastic_grad(params)
        #objective_grad = lambda params: self.stochastic_grad_mini_batch(params, 10)
        #objective_grad = lambda params: self.stochastic_grad_manual(params)
        objective_grad = lambda params: self.stochastic_grad_manual_mini_batch(params, np.min([50,self.N_samples]))
        self.Adadelta(Niterat, objective_grad, 0.05, self.parameters, output_grad_penalty = optimize_penalty)

    def log_posterior_time_shift(self, params, params_time_shift):
        # Input: X, Y and a biomarker's parameters, current time-shift estimates
        # Output: log-posterior and time-shift gradient
        loglik =  0
        Gradient = []
        for l in range(2):
            Gradient.append(np.zeros(self.N_samples))

        # Shifting data according to current time-shift estimate
        for i in range(self.N_biom):
            Xdata = np.array([[1e10]])
            Ydata = np.array([[1e10]])
            for sub in range(self.N_samples):
                temp = self.X_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])):np.sum(self.N_obs_per_sub[i][:sub+1])]
                shifted_temp = (temp * params_time_shift[1][sub] + params_time_shift[0][sub])
                Xdata = np.hstack([Xdata,shifted_temp.T])
                tempY = self.Y_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])):np.sum(self.N_obs_per_sub[i][:sub + 1])]
                Ydata = np.hstack([Ydata, tempY.T])

            Xdata = Xdata[0][1:].reshape([len(Xdata[0][1:]),1])
            Ydata = Ydata[0][1:].reshape([len(Ydata[0][1:]), 1])

            s_omega, m_omega, s_w, m_w, sigma, l, eps = self.unpack_parameters(params[i])
            s_omega = np.exp(s_omega)
            s_w = np.exp(s_w)
            l = np.exp(l)
            sigma = np.exp(sigma)
            eps = np.exp(eps)

            perturbation_zero_W = np.zeros(int(2 * self.N_rnd_features)).reshape([2 * self.N_rnd_features, 1])
            W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s_w))) + m_w
            Omega = 1 / np.sqrt(l) * self.perturbation_Omega

            output = self.basis(Xdata, sigma, Omega)
            Doutput_time_shift = self.Dbasis_time_shift(Xdata, sigma, Omega)

            Doutput = self.Dbasis(self.DX, sigma, Omega)
            Kullback_Leibler = self.KL(s_omega, m_omega, s_w, m_w, l)
            Dterm = np.sum(
                np.log(2) - self.penalty[i] * np.dot(Doutput, W) / 2 + (self.penalty[i] * np.dot(Doutput, W)) ** 2 / 8)
            prior = (eps - 0.3) ** 2 / 1e-2 + (sigma - 0.5) ** 2 / 1e-2   # + (l - np.log(0.2))**2/1e-0
            prior_time_shift = np.sum((params_time_shift[0] - 0)**2/1e-0)

            loglik = loglik - 0.5 * (
            np.log(2 * np.pi * eps) + np.sum((self.Y_array[i] - np.dot(output, W)) ** 2) / eps) - prior - Dterm - Kullback_Leibler - prior_time_shift

            temp = np.multiply(Doutput_time_shift, np.concatenate([Omega , Omega ]))
            grad0 = (((Ydata - np.dot(output, W))) / eps * np.dot(temp, W)).flatten()
            temp = np.multiply(Doutput_time_shift, np.concatenate([Omega * Xdata,Omega * Xdata],1))
            grad1 = (((Ydata - np.dot(output, W))) / eps *  np.dot(temp, W)).flatten()

            for sub in range(self.N_samples):
                temp0 = np.sum([grad0[k] for k in range(int(np.sum(self.N_obs_per_sub[i][:sub])),np.sum(self.N_obs_per_sub[i][:sub+1]))]) - 2 * ((params_time_shift[0] - 0) / 1e-0)[sub]
                temp1 = np.sum([grad1[k] for k in range(int(np.sum(self.N_obs_per_sub[i][:sub])),np.sum(self.N_obs_per_sub[i][:sub+1]))])
                Gradient[0][sub] = Gradient[0][sub] + temp0
                Gradient[1][sub] = Gradient[1][sub] + 0 #temp1

        return loglik, Gradient

    def grad_time_shift(self, params_time_shift):
        output_loglik = []
        objective_cost_function = lambda params_time_shift: \
                    self.log_posterior_time_shift(self.parameters, params_time_shift)

        loglik, MC_grad = objective_cost_function(params_time_shift)

        return loglik, MC_grad


    def Optimize_time_shift(self, Niterat = 10, learning_rate = 0.1):
        # Adadelta for optimization of time shift parameters
        params_time_shift = self.params_time_shift.copy()
        params_time_shift[0] = np.zeros(len(params_time_shift[0]))
        diag = []

        for l in range(2):
            diag.append(np.zeros(len(params_time_shift [l])))
        epsilon = 1e-8

        tot_lik = 0
        gradient = []

        for i in range(Niterat):
            objective_grad = lambda test_params_time_shift: self.grad_time_shift(test_params_time_shift)
            fun_value, fun_grad = objective_grad(params_time_shift)

            tot_lik = fun_value
            for l in range(2):
                diag[l] = 0.9 * diag[l] + 0.1 * fun_grad[l] ** 2
                params_time_shift[l] = params_time_shift[l] + np.multiply(learning_rate * fun_grad[l],
                                                                          1 / np.sqrt(diag[l] + epsilon))
            print i, -tot_lik

        for l in range(1):
            self.params_time_shift[l] = self.params_time_shift[l] + params_time_shift[l]

        for i in range(self.N_biom):
            Xdata = np.array([[100]])
            for sub in range(self.N_samples):
                temp = self.X_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])):np.sum(self.N_obs_per_sub[i][:sub+1])]
                shifted_temp = (temp + params_time_shift[0][sub])
                Xdata = np.hstack([Xdata,shifted_temp.T])

            self.X_array[i] = Xdata[0][1:].reshape([len(Xdata[0][1:]),1])


        self.minX = np.float(np.min([el for sublist in self.X_array for item in sublist for el in item]))
        self.maxX = np.float(np.max([el for sublist in self.X_array for item in sublist for el in item]))
        self.DX = np.linspace(self.minX, self.maxX, self.N_Dpoints).reshape([self.N_Dpoints, 1])


    def Optimize(self, N_global_iterations = 5, iterat = [50,30], Plot = True):
        # Global optimizer (GP parameters + time shift)
        for i in range(N_global_iterations):
            print "iteration ", i, "of ", N_global_iterations
            print "Optimizing GP parameters"
            if i>float(N_global_iterations)-2:
                self.Optimize_GP_parameters(Niterat = iterat[0])
            else:
                self.N_Dpoints = 20
                self.DX = np.linspace(self.minX, self.maxX, self.N_Dpoints).reshape([self.N_Dpoints, 1])
                self.Optimize_GP_parameters(Niterat=iterat[0], optimize_penalty = False)
                #print "Current penalty parameters: "
                #print self.penalty
            if Plot:
                self.Plot()
            if i<(N_global_iterations -1):
                print "Optimizing time shift"
                self.Optimize_time_shift(Niterat=iterat[1])

    def Plot(self, list_biom = []):
        if len(list_biom)<1:
            list_biom = range(self.N_biom)
        # Plot method
        newX = np.linspace(self.minX, self.maxX, 30).reshape([30, 1])
        fig = plt.figure()
        max_y = np.max([np.float(item) for sublist in self.Y_array for item in sublist])
        min_y = np.min([np.float(item) for sublist in self.Y_array for item in sublist])

        for bio_pos,biomarker in enumerate(list_biom):
            if len(list_biom)>3:
                ax = fig.add_subplot((len(list_biom) - 1) / 3 + 1, 3, (bio_pos) + 1)
            else:
                ax = fig.add_subplot(1, len(list_biom) , (bio_pos) + 1)

            if len(self.names_biomarkers)>0:
                plt.title(self.names_biomarkers[biomarker])

            s_omega, m_omega, s, m, sigma, l, eps = self.unpack_parameters(self.parameters[biomarker])


            scaleX = self.max_X[biomarker]*self.mean_std_X[biomarker][1]
            scaleY = self.max_Y[biomarker]*self.mean_std_Y[biomarker][1]

            perturbation_zero_W = np.zeros(int(2 * self.N_rnd_features)).reshape([2 * self.N_rnd_features, 1])
            Omega = 1 / np.sqrt(np.exp(l)) * self.perturbation_Omega
            W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s))) + m
            output = self.basis(newX, np.exp(sigma), Omega)
            predict = np.dot(output, W)

            ax.plot(scaleX * newX +self.mean_std_X[biomarker][0], \
                     predict , lw=2, color='black')

            ax.set_ylim(min_y - 0.5, max_y + 0.5)
            #plt.scatter(self.X_array[biomarker], self.Y_array[biomarker], lw=2, color='red')

            for sub in range(self.N_samples):
                x_data = np.array([self.X_array[biomarker][k][0] for k in range(int(np.sum(self.N_obs_per_sub[biomarker][:sub])), np.sum(self.N_obs_per_sub[biomarker][:sub + 1]))])
                y_data = np.array([self.Y_array[biomarker][k][0] for k in
                         range(int(np.sum(self.N_obs_per_sub[biomarker][:sub])), np.sum(self.N_obs_per_sub[biomarker][:sub + 1]))])
                if len(self.group)>0:
                    col = cm.tab10(self.group[sub])
                    ax.plot(scaleX * x_data + self.mean_std_X[biomarker][0], \
                            y_data, color=col, lw=1)
                else:
                    ax.plot(scaleX * x_data + self.mean_std_X[biomarker][0], \
                          y_data, color='green', lw = 1)

            for i in range(500):
                perturbation_zero_W = np.random.randn(int(2 * self.N_rnd_features)).reshape([ 2 * self.N_rnd_features, 1])
                perturbation_zero_Omega = np.random.randn(int(self.N_rnd_features))
                Omega = 1 / np.sqrt(np.exp(l)) * self.perturbation_Omega
                W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s))) + m
                output = self.basis(newX, np.exp(sigma), Omega)
                predict = np.dot(output, W)

                ax.plot(scaleX * newX +self.mean_std_X[biomarker][0], \
                         predict , lw=0.05, color='red')


        plt.show()

    def Return_time_shift(self):
        individual_time = []
        for sub in range(self.N_samples):
            individual_time.append(np.array([self.X_array[0][k][0] for k in
                               range(int(np.sum(self.N_obs_per_sub[0][:sub])),
                                     np.sum(self.N_obs_per_sub[0][:sub + 1]))])[0])

        scaleX = self.max_X[0] * self.mean_std_X[0][1]
        return np.array(individual_time) *  scaleX + self.mean_std_X[0][0]


    # def Predict(self,test_data):
    #
    #     X_test = test_data[0]
    #     Y_test = test_data[1]
    #     Xrange = np.linspace(self.minX, self.maxX, 30).reshape([30, 1])
    #
    #     pred_sub = []
    #     expectation_sub = []
    #     for sub in range(len(X_test[0])):
    #         print "predicting sub: ", sub, "out of ", len(X_test[0])
    #         pred_sub.append([])
    #         expectation_sub.append([])
    #         for pos_index,position in enumerate(Xrange):
    #             pred_sub[sub].append(0)
    #             for biomarker in range(self.N_biom):
    #                 Y_test_biom = np.array(Y_test[biomarker][sub]).reshape([len(Y_test[biomarker][sub]),1])
    #                 X_test_biom = np.array(X_test[biomarker][sub]).reshape([len(X_test[biomarker][sub]),1])
    #
    #                 X_test_biom = (X_test_biom - self.mean_std_X[biomarker][0]) / self.mean_std_X[biomarker][1]
    #                 X_test_biom = X_test_biom / self.max_X[biomarker]
    #
    #                 Y_test_biom = (Y_test_biom - self.mean_std_Y[biomarker][0]) / self.mean_std_Y[biomarker][1]
    #                 Y_test_biom = Y_test_biom / self.max_Y[biomarker]
    #
    #                 if len(X_test_biom > 0):
    #                     s_omega, m_omega, s, m, sigma, l, eps = self.unpack_parameters(self.parameters[biomarker])
    #                     X_to_test = position + X_test_biom
    #
    #                     #perturbation_zero_W = np.random.randn(int(2 * self.N_rnd_features)).reshape([ 2 * self.N_rnd_features, 1])
    #                     Omega = 1 / np.sqrt(np.exp(l)) * self.perturbation_Omega
    #                     #W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s))) + m
    #                     W = m
    #                     output = self.basis(X_to_test, np.exp(sigma), Omega)
    #                     predict = np.dot(output, W)
    #                     pred_sub[sub][pos_index] = pred_sub[sub][pos_index] - 0.5 * (np.log(2 * np.pi * np.exp(eps)) + np.sum((Y_test_biom - predict) ** 2) / np.exp(eps))
    #
    #         pred_sub[sub] = np.exp(pred_sub[sub])/ np.sum(np.exp(pred_sub[sub]))
    #         scaling = self.mean_std_X[biomarker][1]*self.max_X[biomarker]
    #         expectation_sub[sub] = np.sum(pred_sub[sub] * Xrange.flatten()) * scaling + self.mean_std_X[biomarker][0]
    #     return pred_sub, expectation_sub
    #
    #
    # def Plot_predictions(self,predictions, names = []):
    #     Xrange = np.linspace(self.minX, self.maxX, 30).reshape([30, 1])
    #     for i in range(len(predictions)):
    #         plt.plot(Xrange,predictions[i])
    #         if len(names)>0:
    #             max = np.where(predictions[i]==np.max(predictions[i]))[0]
    #             plt.annotate(names[i],xy = (Xrange[max],predictions[i][max]))
    #     plt.show()

    def Predict(self,test_data):

        X_test = test_data[0]
        Y_test = test_data[1]
        Xrange = np.linspace(self.minX, self.maxX, 30).reshape([30, 1])

        pred_sub = []
        expectation_sub = []

        sampling_dist = []

        for biomarker in range(self.N_biom):
            sampling_dist.append([])
            for i in range(500):
                s_omega, m_omega, s, m, sigma, l, eps = self.unpack_parameters(self.parameters[biomarker])
                perturbation_zero_W = np.random.randn(int(2 * self.N_rnd_features)).reshape(
                    [2 * self.N_rnd_features, 1])
                perturbation_zero_Omega = np.random.randn(int(self.N_rnd_features))
                Omega = 1 / np.sqrt(np.exp(l)) * self.perturbation_Omega
                W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s))) + m
                output = self.basis(Xrange, np.exp(sigma), Omega)
                sampling_dist[biomarker].append(np.dot(output, W))

        for sub in range(len(X_test[0])):
            print "predicting sub: ", sub, "out of ", len(X_test[0])
            pred_sub.append([])
            expectation_sub.append([])
            for pos_index,position in enumerate(Xrange):
                pred_sub[sub].append(0)
                for biomarker in range(self.N_biom):
                    Y_test_biom = np.array(Y_test[biomarker][sub]).reshape([len(Y_test[biomarker][sub]),1])
                    X_test_biom = np.array(X_test[biomarker][sub]).reshape([len(X_test[biomarker][sub]),1])

                    X_test_biom = (X_test_biom - self.mean_std_X[biomarker][0]) / self.mean_std_X[biomarker][1]
                    X_test_biom = X_test_biom / self.max_X[biomarker]

                    Y_test_biom = (Y_test_biom - self.mean_std_Y[biomarker][0]) / self.mean_std_Y[biomarker][1]
                    Y_test_biom = Y_test_biom / self.max_Y[biomarker]

                    if len(X_test_biom > 0):
                        X_to_test = position + X_test_biom
                        for i in range(500):
                            current_sample = sampling_dist[biomarker][i][pos_index:(pos_index + len(Y_test_biom))]
                            if (len(Y_test_biom) == len(current_sample)):
                                pred_sub[sub][pos_index] = pred_sub[sub][pos_index] \
                                                    + np.sum((Y_test_biom - current_sample) ** 2)
                            # - 0.5 * (np.log(2 * np.pi * np.exp(eps)) \

        final_pred = []
        for sub in range(len(pred_sub)):
            valid_indices = np.where(np.array(pred_sub[sub])!=0)
            predictions = np.array(pred_sub[sub])[valid_indices]
            final_pred.append([])
            final_pred[sub] = np.exp(-predictions/500)/ np.sum(np.exp(-predictions/500))
            scaling = self.mean_std_X[biomarker][1]*self.max_X[biomarker]
            #expectation_sub[sub] = np.sum(final_pred[sub] * Xrange.flatten()[valid_indices]) * scaling + self.mean_std_X[biomarker][0]
            expectation_sub[sub] = np.sum(final_pred[sub] * (Xrange.flatten()[valid_indices] * scaling + self.mean_std_X[biomarker][0]))
        return final_pred, expectation_sub


    def Plot_predictions(self,predictions, names = [], group = [], group_name = []):
        Xrange = np.linspace(self.minX, self.maxX, 30).reshape([30, 1])
        scaling = self.mean_std_X[0][1] * self.max_X[0]
        for i in range(len(predictions)):
            valid_indices = np.where(np.array(predictions[i]) != 0)
	    if len(group)>0:
                col_group_sub = cm.tab10(group[i])
                plt.plot(Xrange[valid_indices]* scaling + self.mean_std_X[0][0], np.array(predictions[i])[valid_indices],color=col_group_sub)
            else:
                plt.plot(Xrange[valid_indices]* scaling + self.mean_std_X[0][0], np.array(predictions[i])[valid_indices])
            if len(names)>0:
                max = np.int(np.where(predictions[i]==np.max(predictions[i]))[0])
                plt.annotate(names[i],xy = (Xrange[max]* scaling + self.mean_std_X[0][0],predictions[i][max]))
        plt.title('Predicted staging')
        plt.show()


    def Save(self, path):
        np.save(path + "/0", self.names_biomarkers)
        np.save(path + "/1", self.N_rnd_features)
        np.save(path + "/2", self.N_biom)
        np.save(path + "/3", self.X_array)
        np.save(path + "/4", self.Y_array)
        np.save(path + "/5", self.DX)
        np.save(path + "/6", self.group)
        np.save(path + "/7", self.init_params_full)
        np.save(path + "/8", self.init_params_var)
        np.save(path + "/9", self.max_X)
        np.save(path + "/10", self.max_Y)
        np.save(path + "/11", self.mean_std_X)
        np.save(path + "/12", self.mean_std_Y)
        np.save(path + "/13", self.N_biom)
        np.save(path + "/14", self.N_Dpoints)
        np.save(path + "/15", self.N_rnd_features)
        np.save(path + "/16", self.N_samples)
        np.save(path + "/17", self.parameters)
        np.save(path + "/18", self.params_time_shift)
        np.save(path + "/19", self.penalty)
        np.save(path + "/20", self.perturbation_Omega)
        np.save(path + "/21", self.Y_array)
        np.save(path + "/22", self.maxX)
        np.save(path + "/23", self.minX)

    def Load(self, path):
        self.names_biomarkers = np.load(path + "/0.npy")
        self.N_rnd_features = np.load(path + "/1.npy")
        self.N_biom = np.load(path + "/2.npy")
        self.X_array = np.load(path + "/3.npy")
        self.Y_array = np.load(path + "/4.npy")
        self.DX = np.load(path + "/5.npy")
        self.group = np.load(path + "/6.npy")
        self.init_params_full = np.load(path + "/7.npy")
        self.init_params_var = np.load(path + "/8.npy")
        self.max_X = np.load(path + "/9.npy")
        self.max_Y = np.load(path + "/10.npy")
        self.mean_std_X = np.load(path + "/11.npy")
        self.mean_std_Y = np.load(path + "/12.npy")
        self.N_biom = np.load(path + "/13.npy")
        self.N_Dpoints = np.load(path + "/14.npy")
        self.N_rnd_features = np.load(path + "/15.npy")
        self.N_samples = np.load(path + "/16.npy")
        self.parameters = np.load(path + "/17.npy")
        self.params_time_shift = np.load(path + "/18.npy")
        self.penalty = np.load(path + "/19.npy")
        self.perturbation_Omega = np.load(path + "/20.npy")
        self.Y_array = np.load(path + "/21.npy")
        self.maxX = np.load(path + "/22.npy")
        self.minX = np.load(path + "/23.npy")


def convert_csv(file, list_biomarkers):
    table = pd.read_csv(file)
    X = []
    Y = []

    # list of individuals
    list_RID = np.unique(table[['RID']])

   # list of biomarkers
#    list_biomarkers = table.columns[range(2, len(table.columns))]

    RID = []

    for id_biom, biomarker in enumerate(list_biomarkers):
        X.append([])
        Y.append([])

    # Parsing every biomarker and assigning to the list
    for id_sub, sub in enumerate(list_RID):
        flag_missing = 0
        for id_biom, biomarker in enumerate(list_biomarkers):
            indices = np.where(np.in1d(table.RID, sub))[0]
            X[id_biom].append(np.array(table[['Time']])[np.where(np.in1d(table.RID, sub))[0]].flatten())
            Y[id_biom].append(np.array(table[[biomarker]])[np.where(np.in1d(table.RID, sub))[0]].flatten())

            idx_to_remove = ~np.isnan(Y[id_biom][id_sub])

            Y[id_biom][id_sub] = Y[id_biom][id_sub][idx_to_remove]
            X[id_biom][id_sub] = X[id_biom][id_sub][idx_to_remove]

            if len(Y[id_biom][id_sub]) < 1:
                flag_missing = flag_missing + 1

        if flag_missing == 0:
            RID.append(sub)

    Xtrain = []
    Ytrain = []

    for id_biom, biomarker in enumerate(list_biomarkers):
        Xtrain.append([])
        Ytrain.append([])

    for id_sub, sub in enumerate(list_RID):
        if np.in1d(sub, RID)[0]:
            for id_biom, biomarker in enumerate(list_biomarkers):
                Xtrain[id_biom].append(X[id_biom][id_sub])
                Ytrain[id_biom].append(Y[id_biom][id_sub])

    group = []

    # check if group information is present
    if len(np.where('group' == table.columns)[0])>0:
	for sub in RID:
            group.append(table.group[np.where(np.in1d(table.RID,sub))[0][0]])

    group = np.array(group)

    return Xtrain, Ytrain, RID, list_biomarkers, group
