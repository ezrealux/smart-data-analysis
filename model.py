import torch
import torch.nn as nn
import torch.functional as F

class BSModel(nn.Module):
    def init(self, y_atm):
        super(BSModel, self).__init__()
        self.y_atm = y_atm

    def forward(self, ttm):
        return self.y_atm(ttm)

class SSVIModel(nn.Module):
    def init(self, y_atm, device='cpu', phi_fun='power_law'):
        super(SSVIModel, self).__init__()
        self.y_atm = y_atm
        self.device = device
        self.phi_fun = phi_fun

        self.raw_rho = nn.parameter(torch.tensor([0.5]))
        if self.phi_fun == 'heston_like':
            self.raw_lambda = nn.parameter(torch.tensor([0]))
        elif self.phi_fun == 'power_law':
            self.raw_eta = nn.parameter(torch.tensor([0]))
            self.raw_gamma = nn.parameter(torch.tensor([0]))

    def forward(self, logm, ttm): # (invm, tau): (inverse moneyness, time to maturity)
        y_atm_tau = self.y_atm(ttm)
        rho = torch.tanh(self.raw_rho)

        if self.phi_fun == 'heston_like':
            lambdaa = torch.exp(self.raw_lambda)
            phi = 1 / (lambdaa * y_atm_tau) * (1 - (1 - torch.exp(-lambdaa * self.y_atm)) / (lambdaa * y_atm_tau))
        elif self.phi_fun == 'power_law':
            eta = torch.exp(self.raw_eta)
            gamma = torch.exp(self.raw_gamma)
            phi = eta / (torch.pow(y_atm_tau, gamma) * torch.pow(1 + y_atm_tau, 1 - gamma))

        output = self.y_atm/2 * (1 + rho*phi*logm + torch.sqrt(torch.square(phi*logm) + 2*rho*phi*logm + 1))
        return output

class SmileModel(nn.Module):
    def init(self, hidden_sizes, device='cpu', activation='softplus'):
        super(SmileModel, self).__init__()
        self.device = device
        self.J = hidden_sizes[0]
        self.weights_logm = nn.parameter(nn.init.normal_(torch.empty(self.J, 2, dtype=torch.float64), mean=0, std=0.01))
        self.bias_logm = nn.parameter(nn.init.normal_(torch.empty(self.J, 2, dtype=torch.float64), mean=0, std=0.01))
        self.weights_ttm = nn.parameter(nn.init.normal_(torch.empty(self.J, 2, dtype=torch.float64), mean=0, std=0.01))
        self.bias_ttm = nn.parameter(nn.init.normal_(torch.empty(self.J, 2, dtype=torch.float64), mean=0, std=0.01))
        self.weights_exp = nn.parameter(nn.init.normal_(torch.empty(self.K, 2, dtype=torch.float64), mean=0, std=0.01))
        self.bias_exp = nn.parameter(nn.init.normal_(torch.empty(self.K, 2, dtype=torch.float64), mean=0, std=0.01))

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1])] for i in range(hidden_sizes-1))
        self.output_layers = nn.Linear(hidden_sizes[-1], 1)
        
        if activation == 'tanh':
            self.activation = torch.tanh

    def forward(self, logm, ttm): # (invm, tau): (inverse moneyness, time to maturity)
        smile_function = lambda logm : torch.sqrt(logm*nn.Tanh(logm+0.5) + nn.Tanh(-logm/2+1e-6))

        term_logm = smile_function(self.bias_logm + logm*torch.exp(self.weights_logm)) #element-wise multiplication and addition
        term_ttm = torch.sigmoid(self.bias_ttm + ttm*torch.exp(self.weights_ttm)) #element-wise multiplication and addition
        out_hidden = term_logm * term_ttm * torch.exp(self.weights_exp) + torch.exp(self.bias_exp)
    
        for hidden_layer in self.hidden_layers:
            out_hidden = self.activation(hidden_layer(out_hidden))
        
        output = self.output_layers(out_hidden)
        return output

class NNModel(nn.Module):
    def init(self, input_size, hidden_sizes, device='cpu', activation='softplus'):
        super(NNModel, self).__init__()
        self.device = device
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1])] for i in range(hidden_sizes-1))
        
        if activation == 'tanh':
            self.activation = torch.tanh
                                      
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x

class SingleModel(nn.Module):
    def init(self, input_size, hidden_sizes, prior='Smile', device='cpu'):
        super(SingleModel, self).__init__()
        self.device = device
        if prior == 'BS':
            self.Prior = BSModel()
        elif prior == 'SSVI':
            self.Prior = SSVIModel()
        elif prior == 'Smile':
            self.Prior = SmileModel()
        self.NN = NNModel(input_size, hidden_sizes)

    def forward(self, invm, tau):
        output_Prior = self.Prior(invm, tau)
        output_NN = self.NN(invm, tau)
        output = output_Prior * output_NN
        return output
    
class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.sigmoid = nn.Sigmoid()  # sigma_2
        self.in_hid_weights = nn.Parameter(nn.init.normal_(torch.empty(self.K, 2, dtype=torch.float64), mean=0, std=0.01))
        self.in_hid_bias = nn.Parameter(nn.init.normal_(torch.empty(self.K, 1, dtype=torch.float64), mean=0, std=0.01))
        self.hid_out_weights = nn.Parameter(nn.init.normal_(torch.empty(self.K, self.I, dtype=torch.float64), mean=0, std=0.01))
        self.hid_out_bias = nn.Parameter(nn.init.normal_(torch.empty(self.I, dtype=torch.float64), mean=0, std=0.01))

    def forward(self, logm, ttm):
        weight_numerator = self.sigmoid((self.in_hid_weights @ torch.vstack((logm, ttm))) + self.in_hid_bias)  # (K,2)*(2*1)=(K,1)
        weight_denominator = torch.exp(weight_numerator.T @ self.hid_out_weights + self.hid_out_bias)  # (1,K)*(K*I)=(1,I)
        weight_denominator = torch.sum(weight_denominator, axis=1) # shape = scalar

        return weight_denominator

class MultiModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, ensemble_num=5, device='cpu'):
        super(MultiModel, self).__init__()
        self.device = device
        self.ensemble_num = ensemble_num # the number of single models in Multi
        self.ensemble_list = nn.ModuleList()

        for _ in range(self.ensemble_num):
            self.single_model_list.append(SingleModel(input_size, hidden_sizes))

    def forward(self, logm, ttm):
        outputs = []
        for i in range:
            outputs[i] = self.single_model_list(logm, ttm)
        weights = self.SoftmaxModel(logm, ttm)
        
        output = torch.dot(outputs, weights)
        return output
    
class WeightedSumLoss(nn.Module):
    def __init__(self, losses, weights):
        super(WeightedSumLoss, self).__init__()
        if len(losses) != len(weights):
            raise ValueError("Number of losses and weights must be the same.")
        self.RMSELoss = RMSELoss()
        self.MAPELoss = MAPELoss()
        self.CalenderLoss = Loss_calendar()
        self.ButterflyLoss = Loss_butterfly()
        self.LinearLoss = Loss_linear()
        self.UpperBoundLoss = Loss_upperbound()
        self.weights = torch.IntTensor(weights)

    def forward(self, y_pred, y_true, logm, ttm):
        losses = torch.FloatTensor([self.RMSELoss(y_pred, y_true),
                                    self.MAPELoss(y_pred, y_true),
                                    self.CalenderLoss(y_pred, y_true, ttm),
                                    self.ButterflyLoss(y_pred, y_true, logm),
                                    self.LinearLoss(y_pred, y_true, logm),
                                    self.UpperBoundLoss(y_pred, y_true, logm)])
        Total_loss = torch.dot(self.weights, losses)
        return Total_loss, losses
    
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
        
        return rmse

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-6
        absolute_percentage_error = torch.abs((y_true - y_pred) / (y_true + epsilon))
        mape = torch.mean(absolute_percentage_error)
        
        return mape
    
class Loss_calendar:
    def __init__(self):
        super(Loss_calendar, self).__init__()

    def forward(self, y_pred, ttm):
        ttm.requires_grad = True
        neg_dy_dt = -1*torch.autograd.grad(y_pred, ttm, create_graph=True, retain_graph=True)[0]

        loss = torch.mean(torch.relu(neg_dy_dt))
        return loss
    
class Loss_butterfly:
    def __init__(self) -> None:
        super(Loss_butterfly, self).__init__()

    def forward(self, y_pred, logm):
        logm.requires_grad = True  # Enable gradients for input x
        dy_dlogm = torch.autograd.grad(y_pred, logm, create_graph=True, retain_graph=True)[0]
        d2y_dlogm2 = torch.autograd.grad(dy_dlogm, logm, create_graph=True, retain_graph=True)[0]

        g_k = (1-(logm*dy_dlogm)/(2*y_pred))**2 - dy_dlogm/4*(1/y_pred+0.25) + d2y_dlogm2/2

        loss = torch.mean(torch.relu(g_k))
        return loss
    
class Loss_linear:
    def __init__(self):
        super(Loss_linear, self).__init__()

    def forward(self, y_pred, logm):
        logm.requires_grad = True  # Enable gradients for input x
        dy_dlogm = torch.autograd.grad(y_pred, logm, create_graph=True, retain_graph=True)[0]
        d2y_dlogm2 = torch.autograd.grad(dy_dlogm, logm, create_graph=True, retain_graph=True)[0]

        loss = torch.mean(d2y_dlogm2)
        return loss
    
class Loss_upperbound:
    def __init__(self):
        super(Loss_upperbound, self).__init__()

    def forward(self, y_pred, logm):
        diff = y_pred - torch.abs(2 * logm)

        loss = torch.mean(torch.relu(diff))
        return loss