import torch
from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torchvision.transforms as transforms


class PushTDiffusionDataset(Dataset):
    def __init__(self, dataset_name="lerobot/pusht", horizon=16):
        self.lerobot_ds = LeRobotDataset(dataset_name, video_backend="pyav")
        self.horizon = horizon

        self.stats = self.lerobot_ds.meta.stats

        # Image preprocessing (resnet-18 standard)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.lerobot_ds) - self.horizon

    def __getitem__(self, idx):
        curr_item = self.lerobot_ds[idx]

        # Image (3, 96, 96) -> normalize in [0~1]
        image = curr_item["observation.image"].float()
        image = self.normalize(image)

        raw_state = curr_item["observation.state"].float()
        s_min = self.stats["observation.state"]["min"]
        s_max = self.stats["observation.state"]["max"]
        state = ((raw_state - s_min) / (s_max - s_min + 1e-8) * 2 - 1).float()

        # Action Chunking (t ~ t+H-1)
        actions = []
        for i in range(self.horizon):
            raw_action = self.lerobot_ds[idx + i]["action"]
            # Normalize actions within [-1, 1]
            # (x - min) / (max - min) * 2 - 1
            a_min = self.stats["action"]["min"]
            a_max = self.stats["action"]["max"]
            norm_action = (
                (raw_action - a_min) / (a_max - a_min + 1e-8) * 2 - 1
            ).float()
            actions.append(norm_action)

        action_chunk = torch.stack(actions)  # (Horizon, Action_Dim) -> (16, 2)

        return {
            "image": image,  # (3, 96, 96)
            "agent_pos": state,  # (2)
            "action": action_chunk,  # (16, 2)
        }


def get_dataloader(batch_size=32, horizon=16):
    dataset = PushTDiffusionDataset(horizon=horizon)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
