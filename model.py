"""
@author: Victor Michel-Dansac <victor.michel-dansac@inria.fr>
"""

import copy as cp

import torch
import torch.nn as nn
from torch.autograd import Variable, grad

try:
    import torchinfo
except ModuleNotFoundError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)


class Net(nn.DataParallel):
    def __init__(self, layer_sizes):
        super(Net, self).__init__(nn.Module())

        self.hidden_layers = []
        for l1, l2 in zip(layer_sizes[:-1], layer_sizes[+1:]):
            self.hidden_layers.append(nn.Linear(l1, l2).double())
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output_layer = nn.Linear(layer_sizes[-1], 1).double()

    def forward(self, x, a, b, u0):
        inputs = torch.cat([x, a, b, u0], axis=1)

        layer_output = torch.tanh(self.hidden_layers[0](inputs))
        for hidden_layer in self.hidden_layers[1:]:
            layer_output = torch.tanh(hidden_layer(layer_output))

        return self.output_layer(layer_output)


class Network:
    DEFAULT_X_MIN, DEFAULT_X_MAX = 0, 1
    DEFAULT_A_MIN, DEFAULT_A_MAX = 0.5, 1
    DEFAULT_B_MIN, DEFAULT_B_MAX = 0.5, 1
    DEFAULT_U0_MIN, DEFAULT_U0_MAX = 0.1, 0.2

    DEFAULT_FILE_NAME = "network_advection.pth"

    DEFAULT_LEARNING_RATE = 1e-3

    DEFAULT_LAYER_SIZES = [4, 16, 32, 32, 16, 5]

    def __init__(self, **kwargs):
        self.x_min = kwargs.get("x_min", self.DEFAULT_X_MIN)
        self.x_max = kwargs.get("x_max", self.DEFAULT_X_MAX)

        self.a_min = kwargs.get("a_min", self.DEFAULT_A_MIN)
        self.a_max = kwargs.get("a_max", self.DEFAULT_A_MAX)

        self.b_min = kwargs.get("b_min", self.DEFAULT_B_MIN)
        self.b_max = kwargs.get("b_max", self.DEFAULT_B_MAX)

        self.u0_min = kwargs.get("u0_min", self.DEFAULT_U0_MIN)
        self.u0_max = kwargs.get("u0_max", self.DEFAULT_U0_MAX)

        self.file_name = kwargs.get("file_name", self.DEFAULT_FILE_NAME)

        self.learning_rate = kwargs.get("learning_rate", self.DEFAULT_LEARNING_RATE)

        self.layer_sizes = self.DEFAULT_LAYER_SIZES

        self.create_network()
        self.load(self.file_name)

    @staticmethod
    def u_exact(
        x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, u0: torch.Tensor
    ) -> torch.Tensor:
        """Compute the exact solution.

        Args:
            x: batch of points where to evaluate the solution,
            a: batch of values of the parameter $a$,
            b: batch of values of the parameter $b$,
            u0: batch of values of the parameter $u_0$

        Return:
            exact steady solution of the advection equation, same shape as x
        """
        return a * u0 / ((a + b * u0) * torch.exp(-a * x) - b * u0)

    def get_u(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        u0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the approximation of the steady solution from the model.

        Args:
            self: the neural network $W_θ$,
            x: batch of points where to evaluate the solution,
            a: batch of values of the parameter $a$,
            b: batch of values of the parameter $b$,
            u0: batch of values of the parameter $u_0$


        Return:
            approximation $\\tilde W_θ$ of the steady solution
        """
        return u0 + x * self(x, a, b, u0)

    def residual(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        u0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the residual of the ODE.

        Args:
            self: the neural network $W_θ$,
            x: batch of points where to evaluate the solution,
            a: batch of values of the parameter $a$,
            b: batch of values of the parameter $b$,
            u0: batch of values of the parameter $u_0$,

        Return:
            ODE residual
        """
        u = self.get_u(x, a, b, u0)
        u_x = grad(u, x, torch.ones_like(x), create_graph=True)[0]
        return u_x - a * u - b * u**2

    def pde_loss(self, n_collocation: int) -> torch.Tensor:
        """Compute the PDE loss $\\mathcal{J}_\\text{PDE}$.

        Args:
            self: the neural network $W_θ$,
            n_collocation: the number of collocation points

        Return:
            the PDE loss $\\mathcal{J}_\\text{PDE}$
        """
        shape = (n_collocation, 1)

        x = self.random(self.x_min, self.x_max, shape, requires_grad=True)
        a = self.random(self.a_min, self.a_max, shape, requires_grad=True)
        b = self.random(self.b_min, self.b_max, shape, requires_grad=True)
        u0 = self.random(self.u0_min, self.u0_max, shape, requires_grad=True)

        zeros = torch.zeros(shape)

        residual = self.residual(x, a, b, u0)

        return self.cost_function(residual, zeros)

    def data_loss(self, n_data: int) -> torch.Tensor:
        """Compute the data loss $\\mathcal{J}_\\text{data}$.

        Args:
            self: the neural network $W_θ$,
            n_data: the number of data points

        Return:
            the data loss $\\mathcal{J}_\\text{data}$
        """
        shape = (n_data, 1)

        x = self.random(self.x_min, self.x_max, shape)
        a = self.random(self.a_min, self.a_max, shape)
        b = self.random(self.b_min, self.b_max, shape)
        u0 = self.random(self.u0_min, self.u0_max, shape)

        u = self.u_exact(x, a, b, u0)

        u_pred = self.get_u(x, a, b, u0)

        return self.cost_function(u_pred, u)

    @staticmethod
    def cost_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the MSE cost function between the two inputs.

        Args:
            x, tensor
            y, tensor

        Return:
            float, the MSE cost function between x and y: `||x - y||₂²`
        """
        return torch.nn.MSELoss()(x, y)

    def __call__(self, *args):
        return self.net(*args)

    def create_network(self):
        self.net = nn.DataParallel(Net(self.layer_sizes))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def load(self, file_name):
        self.loss_history = []

        try:
            try:
                checkpoint = torch.load(file_name)
            except RuntimeError:
                checkpoint = torch.load(file_name, map_location=torch.device("cpu"))

            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.loss = checkpoint["loss"]

            try:
                self.loss_history = checkpoint["loss_history"]
            except KeyError:
                pass

        except FileNotFoundError:
            print("network was not loaded from file: training needed")

    @staticmethod
    def save(file_name, epoch, net_state, optimizer_state, loss, loss_history):
        torch.save(
            {
                epoch: epoch,
                "model_state_dict": net_state,
                "optimizer_state_dict": optimizer_state,
                "loss": loss,
                "loss_history": loss_history,
            },
            file_name,
        )

    def predict_u_from_floats(
        self, x: torch.Tensor, a: float, b: float, u0: float
    ) -> torch.Tensor:
        ones = torch.ones(x.shape, requires_grad=False)

        a = a * ones
        b = b * ones
        u0 = u0 * ones

        return self.get_u(x, a, b, u0)

    def predict_u_from_numpy(self, x_numpy, mesh):
        """
        Uses the model to predict the value of u at the space points x_numpy,
        using the parameters a, b and u0 contained in the instance of mesh.

        Args:
            x_numpy, a numpy array containing the space points
            mesh, an instance of the Mesh class, containing the values of a, b and u0

        Return:
            a numpy array, of shape x.shape, containing the prediction of u(x)
        """
        reshaped_x = x_numpy.reshape((x_numpy.size, 1))
        x = Variable(torch.from_numpy(reshaped_x).double(), requires_grad=False)

        ones = torch.ones_like(x)

        a = mesh.a * ones
        b = mesh.b * ones
        u0 = mesh.u0 * ones

        u_pred = self.get_u(x, a, b, u0)

        return u_pred.detach().cpu().numpy().reshape(x_numpy.shape)

    def predict_dxu_from_numpy(self, x_numpy, mesh):
        """
        Uses the model to predict the value of du/dx at the space points x_numpy,
        using the parameters a, b and u0 contained in the instance of mesh.

        Args:
            x_numpy, a numpy array containing the space points
            mesh, an instance of the Mesh class, containing the values of a, b and u0

        Return:
            a numpy array, of shape x.shape, containing the prediction of du/dx(x)
        """
        reshaped_x = x_numpy.reshape((x_numpy.size, 1))
        x = Variable(torch.from_numpy(reshaped_x).double(), requires_grad=True)

        ones = torch.ones_like(x)

        a = mesh.a * ones
        b = mesh.b * ones
        u0 = mesh.u0 * ones

        u_pred = self.get_u(x, a, b, u0)
        dxu_pred = grad(u_pred, x, ones, create_graph=False)[0]

        return dxu_pred.detach().cpu().numpy().reshape(x_numpy.shape)

    def __str__(self):
        try:
            return str(
                torchinfo.summary(
                    self.net.module,
                    [(1, 1), (1, 1), (1, 1), (1, 1)],
                    dtypes=[torch.double, torch.double, torch.double, torch.double],
                )
            )
        except NameError:
            return "torchinfo not found, unable to print the model"

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False):
        """
        Computes uniformly sampled points in the interval ('min_value', 'max_value').
        The resulting tensor is of shape 'shape'.

        Args:
            min_value, float, the lower bound of the sampling interval
            max_value, float, the upper bound of the sampling interval
            shape, tuple of ints, the shape of the resulting tensor
            requires_grad, bool, to be set to True if gradients are required

        Return:
            a tensor containing the uniformly sampled points
        """
        random_numbers = torch.rand(shape, requires_grad=requires_grad)
        return min_value + (max_value - min_value) * random_numbers

    def train(self, **kwargs):
        n_epochs = kwargs.get("n_epochs", 500)
        n_collocation = kwargs.get("n_collocation", 10_000)
        n_data = kwargs.get("n_data", 0)

        assert (
            n_data > 0 or n_collocation > 0
        ), "n_data and n_collocation must not both be zero"

        try:
            best_loss_value = self.loss.item()
        except AttributeError:
            best_loss_value = 1e10

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()

            self.loss = 0

            if n_collocation > 0:
                # Loss based on PDE residual
                self.loss += self.pde_loss(n_collocation)

            if n_data > 0:
                # Loss based on data
                self.loss += self.data_loss(n_data)

            self.loss.backward()
            self.optimizer.step()

            self.loss_history.append(self.loss.item())

            if epoch % 500 == 0:
                print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")

            if self.loss.item() < best_loss_value:
                print(f"epoch {epoch: 5d}: best loss = {self.loss.item():5.2e}")
                best_loss = self.loss.clone()
                best_loss_value = best_loss.item()
                best_net = cp.deepcopy(self.net.state_dict())
                best_optimizer = cp.deepcopy(self.optimizer.state_dict())

        print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")

        try:
            self.save(
                self.file_name,
                epoch,
                best_net,
                best_optimizer,
                best_loss,
                self.loss_history,
            )
        except UnboundLocalError:
            pass

        self.plot_result()

    def plot_result(self, random=False, n_plots=1):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(n_plots, 3, figsize=(15, 5 * n_plots))

        if n_plots == 1:
            ax = ax[None, :]

        for i_plot in range(n_plots):
            ax[i_plot, 0].semilogy(self.loss_history)
            ax[i_plot, 0].set_title(f"loss history, epoch={len(self.loss_history)}")

            if random:
                a = self.random(self.a_min, self.a_max, (1,))
                b = self.random(self.b_min, self.b_max, (1,))
                u0 = self.random(self.u0_min, self.u0_max, (1,))
            else:
                a = (self.a_min + self.a_max) / 2
                b = (self.b_min + self.b_max) / 2
                u0 = (self.u0_min + self.u0_max) / 2

            n_visu = 500

            x = torch.linspace(0, 1, n_visu)[:, None]
            u_pred = self.predict_u_from_floats(x, a, b, u0)
            u_exact = self.u_exact(x, a, b, u0)

            ax[i_plot, 1].plot(x.cpu(), u_exact.detach().cpu(), label="exact")
            ax[i_plot, 1].plot(x.cpu(), u_pred.detach().cpu(), label="prediction")

            try:
                a = a.item()
                b = b.item()
                u0 = u0.item()
            except AttributeError:
                pass

            title = rf"prediction, $a$={a:3.2f}, $b$={b:3.2f}, $u_0$={u0:3.2f}"

            ax[i_plot, 1].set_title(title)
            ax[i_plot, 1].legend()

            error = torch.abs(u_pred - u_exact).detach().cpu()

            ax[i_plot, 2].plot(x.cpu(), error)
            ax[i_plot, 2].set_title("prediction error")

        fig.tight_layout()
        plt.show()
