import torch 
import random 
import numpy as np
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf
import os

from modules import get_model
from modules_kvcache import get_causal_kv_model
from scorer import MetricScorer, save_results_npz
from dataset import UTSDataset, GiftEval, SyntheticTimeSeriesDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):

    OmegaConf.set_struct(config, False)
    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(config))
    print(f"---------")

    settings = config.settings
    config_model = config.model
    config_dataset = config.dataset

    test_names = config_dataset.testsets

    for seq_len in [128, 256, 512, 1024]:
        print(f"Running experiments for sequence length: {seq_len}")
        eval_target_len = config_dataset.max_pred_len

        print(f"Preparing datasets...")
        test_datasets = {}
        for name in test_names:

            if name == "utsd":
                print(f"Loading UTSD dataset for evaluation...")
                utsd = UTSDataset(input_len=seq_len, output_len=eval_target_len, stride=128, flag="val", subset_name="UTSD-12G")
                test_datasets["utsd"] = utsd

            elif name == "gift_eval":
                print(f"Loading GiftEval_eval dataset for evaluation...")
                gift_eval = GiftEval(path="data/", input_len=seq_len, output_len=eval_target_len, stride=16)
                test_datasets["gift_eval"] = gift_eval

            elif name == "artificial":
                print(f"Loading SyntheticTimeSeriesDataset for evaluation...")
                artificial = SyntheticTimeSeriesDataset(seq_len=seq_len, target_len=eval_target_len, noise=True, n_samples=1000)
                test_datasets["artificial"] = artificial

        print(f"Datasets ready.")

        for revin_name in ["CausalRevIN", "RevIN", "PrefixRevIN", "PrefixRevIN2"]:
            for use_asinh in [True, False]:
                torch.manual_seed(0)
                random.seed(0)
                np.random.seed(0)

                print(f"Running experiment with {revin_name}, use_asinh={use_asinh}...")

                config_model.revin_config_name = revin_name
                config_model.use_asinh = use_asinh

                if revin_name=="CausalRevIN":
                    print(f"Using causal KV cache model for {revin_name}...")
                    model = get_causal_kv_model(use_asinh=use_asinh, device=DEVICE)
                else:
                    model = get_model(revin_strategy=revin_name, use_asinh=use_asinh, device=DEVICE)
                
                test_loaders = {name: torch.utils.data.DataLoader(
                    dataset,
                    batch_size=settings.batch_size,
                    shuffle=False,
                    num_workers=settings.num_workers,
                    pin_memory=settings.pin_memory
                ) for name, dataset in test_datasets.items()}

                print("Starting testing...")
                for test_name, test_loader in test_loaders.items():
                    print(f"Testing on {test_name} dataset...")
                    
                    scorer = MetricScorer(
                        max_pred_len=eval_target_len,
                        patch_len=config_model.patch_len
                    )
                    
                    with torch.no_grad():
                        for i, batch in enumerate(tqdm.tqdm(test_loader)):
                            x, y = batch
                            x = x.to(DEVICE)
                            y = y.to(DEVICE)
                            prediction = model.forecast(x, target_len=y.size(1))  
                            scorer.update(prediction, y, x)

                    cur_results = scorer.compute()
                    scorer.reset()

                    if not os.path.exists("./raw_results"):
                        os.makedirs("./raw_results")

                    str_dir = f"./raw_results/{config_model.revin_config_name}_{config_model.use_asinh}/{seq_len}"
                    if not os.path.exists(str_dir):
                        os.makedirs(str_dir)
                        
                    save_results_npz(cur_results, f"{str_dir}/results_{test_name}.npz")
                    print(f"Results saved for {test_name} dataset.")
                
                print("Testing completed.")

if __name__ == '__main__':
    main()