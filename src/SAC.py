import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings

# Suppress sklearn UserWarning from LDA when n_components is large
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ==========================================================
# 0. CONFIGURATION (HARDCODED PARAMETERS)
# ==========================================================
class Config:
    """Configuration class to hold all hyperparameters."""
    BATCH_SIZE = 256
    EPOCHS = 15
    LR = 0.01
    MOMENTUM = 0.9
    D_COMPONENTS_LIST = [32, 64, 128]  # Ranks for ablation study
    SEED = 42
    USE_CUDA = torch.cuda.is_available() # Automatically detect CUDA

# ==========================================================
# 1. DATA LOADING & BASE MATRIX GENERATION
# ==========================================================
class DataAndBaseManager:
    """Handles data loading and generation of various base matrices."""
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.train_loader, self.test_loader = self._load_data()
        self.n_input = 32 * 32 * 3
        self.bases = {}
        print("--- Loading data and generating base matrices ---")
        self._generate_bases()

    def _load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, test_loader

    def _generate_bases(self):
        print("Fetching all training data for PCA/LDA...")
        # Note: In very large scale settings, one might use a subset.
        all_train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        # Use a single-batch loader to grab all data at once efficiently
        all_train_loader = DataLoader(all_train_dataset, batch_size=len(all_train_dataset), shuffle=False)
        all_data, all_labels = next(iter(all_train_loader))

        flat_data = all_data.view(-1, self.n_input).numpy()
        labels = all_labels.numpy()

        # 1. PCA Base
        print("Performing PCA...")
        mean = torch.from_numpy(flat_data).mean(dim=0)
        centered_data = torch.from_numpy(flat_data) - mean
        max_rank = min(centered_data.shape)
        _, _, V = torch.svd_lowrank(centered_data, q=max_rank)
        self.bases['pca'] = V.to(self.device)
        print(f"PCA base generated. Shape: {self.bases['pca'].shape}")

        # 2. Random Orthonormal Base
        print("Generating Random Orthonormal base...")
        random_matrix = torch.randn(self.n_input, self.n_input)
        q, _ = torch.linalg.qr(random_matrix)
        self.bases['random'] = q.to(self.device)
        print(f"Random base generated. Shape: {self.bases['random'].shape}")

        # 3. LDA Base
        print("Performing LDA...")
        n_lda_comp = len(np.unique(labels)) - 1
        lda = LDA(n_components=n_lda_comp)
        lda.fit(flat_data, labels)
        lda_base = torch.tensor(lda.scalings_, dtype=torch.float32)
        self.bases['lda'] = lda_base.to(self.device)
        print(f"LDA base generated. Shape: {self.bases['lda'].shape}. (Limited to {n_lda_comp} components by definition)")

    def get_base(self, base_type, d_components):
        if base_type not in self.bases:
            raise ValueError(f"Base type '{base_type}' not recognized.")

        full_base = self.bases[base_type]

        if d_components > full_base.shape[1]:
            print(f"Warning: Requested {d_components} components for '{base_type}', but only {full_base.shape[1]} are available. Using {full_base.shape[1]}.")
            d_components = full_base.shape[1]

        return full_base[:, :d_components]

# ==========================================================
# 2. MODEL DEFINITIONS
# ==========================================================
class SAC_Linear(nn.Module):
    def __init__(self, in_features, out_features, base_matrix):
        super(SAC_Linear, self).__init__()
        self.register_buffer('base', base_matrix)
        d_comp = base_matrix.shape[1]
        self.learner = nn.Parameter(torch.randn(out_features, d_comp))
        nn.init.kaiming_uniform_(self.learner, a=np.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        weight = self.learner @ self.base.T
        return torch.functional.F.linear(x, weight, self.bias)

class SAC_MLP(nn.Module):
    def __init__(self, base_matrix, n_input, n_hidden, n_output):
        super(SAC_MLP, self).__init__()
        self.layers = nn.Sequential(
            SAC_Linear(n_input, n_hidden, base_matrix),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

class StandardMLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(StandardMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

# ==========================================================
# 3. EXPERIMENT RUNNER
# ==========================================================
class ExperimentRunner:
    def __init__(self, data_manager, config, device):
        self.data = data_manager
        self.config = config
        self.device = device
        self.results = []
        self.criterion = nn.CrossEntropyLoss()

    def _train_and_evaluate(self, model, experiment_name):
        model.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config.LR, momentum=self.config.MOMENTUM)

        print(f"\n--- Running Experiment: {experiment_name} ---")
        start_time = time.time()

        best_acc = 0.0
        for epoch in range(self.config.EPOCHS):
            model.train()
            for data, target in self.data.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

            test_acc = self._evaluate(model)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.config.EPOCHS}, Test Accuracy: {test_acc:.2f}%")
            best_acc = max(best_acc, test_acc)

        total_time = time.time() - start_time
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Finished. Best Accuracy: {best_acc:.2f}%, Time: {total_time:.2f}s, Params: {num_params:,}")

        self.results.append({
            'Experiment': experiment_name,
            'Best Accuracy (%)': round(best_acc, 2),
            'Trainable Params': num_params,
            'Training Time (s)': round(total_time, 2),
        })

    def _evaluate(self, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.data.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total

    def run_all(self):
        N_INPUT, N_HIDDEN, N_OUTPUT = self.data.n_input, 512, 10

        # Control Experiment: Baseline
        torch.manual_seed(self.config.SEED)
        model_baseline = StandardMLP(N_INPUT, N_HIDDEN, N_OUTPUT)
        self._train_and_evaluate(model_baseline, "Baseline (Full Rank)")

        # Ablation Study 1: Base Type
        print("\n" + "="*50 + "\nABLATION STUDY 1: EFFECT OF BASE TYPE\n" + "="*50)
        d_comp_for_base_ablation = 128
        for base_type in ['pca', 'random', 'lda']:
            torch.manual_seed(self.config.SEED)
            base_matrix = self.data.get_base(base_type, d_comp_for_base_ablation)
            model_sac = SAC_MLP(base_matrix, N_INPUT, N_HIDDEN, N_OUTPUT)
            exp_name = f"SAC (Base={base_type.upper()}, d={base_matrix.shape[1]})"
            self._train_and_evaluate(model_sac, exp_name)

        # Ablation Study 2: Rank (d_components)
        print("\n" + "="*50 + "\nABLATION STUDY 2: EFFECT OF RANK (d_components)\n" + "="*50)
        for d_comp in self.config.D_COMPONENTS_LIST:
            torch.manual_seed(self.config.SEED)
            base_matrix = self.data.get_base('pca', d_comp)
            model_sac = SAC_MLP(base_matrix, N_INPUT, N_HIDDEN, N_OUTPUT)
            exp_name = f"SAC (Base=PCA, d={d_comp})"
            self._train_and_evaluate(model_sac, exp_name)

    def report_results(self):
        print("\n" + "="*70 + "\n" + " " * 25 + "FINAL EXPERIMENT RESULTS\n" + "="*70)
        if not self.results:
            print("No results to display.")
            return

        df = pd.DataFrame(self.results)
        baseline_params = df.loc[df['Experiment'] == 'Baseline (Full Rank)', 'Trainable Params'].iloc[0]
        df['Param Reduction'] = (1 - df['Trainable Params'] / baseline_params).apply(lambda x: f"{x:.1%}" if x > 0 else "-")
        df = df[['Experiment', 'Best Accuracy (%)', 'Trainable Params', 'Param Reduction', 'Training Time (s)']]
        print(df.to_string(index=False))
        print("="*70)

# ==========================================================
# 4. MAIN EXECUTION
# ==========================================================
if __name__ == '__main__':
    # Instantiate the configuration
    config = Config()

    # Set seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Setup device
    device = torch.device("cuda" if config.USE_CUDA else "cpu")
    if config.USE_CUDA:
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Using device: {device}")

    # 1. Prepare data and bases
    data_manager = DataAndBaseManager(config, device)

    # 2. Run experiments
    runner = ExperimentRunner(data_manager, config, device)
    runner.run_all()

    # 3. Report final results
    runner.report_results()
