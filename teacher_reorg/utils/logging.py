import os
import numpy as np
import wandb


class Logger:
    def __init__(self, root_dir, args):
        self.root_dir = root_dir
        self.args = args
        self.train_loss_history = {"epoch": []}
        self.val_loss_history = {"epoch": []}
        self.test_loss_history = {"epoch": []}
        self.hand_draw_loss_history = {"epoch": []}
        self.lr_history = {"epoch": [], "lr": []}
        self.logging_file = None

        self.setup_logging()
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            self.setup_wandb()

    def setup_logging(self):
        self.logging_file = os.path.join(self.root_dir, 'logging.txt')
        with open(self.logging_file, 'w') as f:
            f.write("Training Loss, Validation Loss, Learning Rate\n")

    def setup_wandb(self):
        project = self.args.project
        entity = "sketch_rl"
        log_dir = os.path.join(self.root_dir, 'wandb')
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project=project, entity=entity, name=self.args.unique_token, config=self.args, dir=log_dir)

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

        wandb.define_metric("val/step")
        wandb.define_metric("val/*", step_metric="val/step")
        
        wandb.define_metric("avg_epoch/step")
        wandb.define_metric("avg_epoch/*", step_metric="avg_epoch/step")

        wandb.define_metric("test/step")
        wandb.define_metric("test/*", step_metric="test/step")

        wandb.define_metric("hand_draw/step")
        wandb.define_metric("hand_draw/*", step_metric="hand_draw/step")

    def log_wandb(self, log_dict):
        if self.use_wandb:
            wandb.log(log_dict)

    def finish_wandb(self):
        wandb.finish()

    def get_cur_loss(self, loss_type):
        if loss_type == 'train':
            cur_loss = self.train_loss_history
            prefix = 'train'
        elif loss_type == 'val':
            cur_loss = self.val_loss_history
            prefix = 'val'
        elif loss_type == 'test':
            cur_loss = self.test_loss_history
            prefix = 'test'
        elif loss_type == 'hand_draw':
            cur_loss = self.hand_draw_loss_history
            prefix = 'hand_draw'
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        
        return cur_loss, prefix
    
    def log_loss(self, epoch, loss, loss_type):
        cur_loss, prefix = self.get_cur_loss(loss_type)
        
        if prefix == 'train' or prefix == 'val':
            cur_loss["epoch"].append(epoch)

        log_dict = {
            f"{prefix}/step": len(cur_loss[list(cur_loss.keys())[0]]) - 1,
        }

        for key in loss.keys():
            if key not in cur_loss:
                cur_loss[key] = []
            cur_loss[key].append(loss[key].item())

            log_dict[f'{prefix}/{key}'] = loss[key].item()
        
        self.log_wandb(log_dict)
        
    def log_lr(self, epoch, lr):
        self.lr_history["epoch"].append(epoch)
        self.lr_history["lr"].append(lr)

        log_dict = {
            "train/step": len(self.lr_history["lr"]) - 1,
            'train/lr': lr
        }
        self.log_wandb(log_dict)

    def log_epoch_loss_to_file(self, epoch):
        idx = np.array(self.train_loss_history["epoch"]) == epoch
        train_loss = {key: np.mean(np.array(self.train_loss_history[key])[idx]) for key in self.train_loss_history if key != "epoch"}
        idx = np.array(self.val_loss_history["epoch"]) == epoch
        val_loss = {key: np.mean(np.array(self.val_loss_history[key])[idx]) for key in self.val_loss_history if key != "epoch"}
        idx = np.array(self.lr_history["epoch"]) == epoch
        lr = np.mean(np.array(self.lr_history["lr"])[idx])
        
        # Log to wandb
        log_dict = {"avg_epoch/step": epoch}
        for key in train_loss.keys():
            log_dict[f'avg_epoch/train_{key}'] = train_loss[key]
        for key in val_loss.keys():
            log_dict[f'avg_epoch/val_{key}'] = val_loss[key]
        log_dict['avg_epoch/lr'] = lr
        self.log_wandb(log_dict)    

        # Log to text file
        message = [f"{key}: {train_loss[key]:.4f}" for key in train_loss]
        message = ", ".join(message)
        message = f"Epoch {epoch+1}/{self.args.num_epochs}, train, {message}, Current LR: {lr}"
        print(message)
        with open(self.logging_file, 'a') as f:
            f.write(f"{message}\n")

        message = [f"{key}: {val_loss[key]:.4f}" for key in val_loss]
        message = ", ".join(message)
        print(f"Epoch {epoch+1}/{self.args.num_epochs}, val, {message}\n")
        with open(self.logging_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{self.args.num_epochs}, val, {message}\n\n")

    def log_test_loss_to_file(self, loss_type):
        cur_loss, prefix = self.get_cur_loss(loss_type)
        len_loss = len(cur_loss[list(cur_loss.keys())[0]])
        for i in range(len_loss):
            message = [f"{key}: {cur_loss[key][i]:.4f}" for key in cur_loss]
            message = ", ".join(message)
            message = f"{prefix} {i+1}/{len_loss}, {message}\n"
            with open(self.logging_file, 'a') as f:
                f.write(message)
        
        ave_message = [f"{key}: {np.mean(cur_loss[key]):.4f}" for key in cur_loss]
        ave_message = ", ".join(ave_message)
        ave_message = f"{prefix} average, {ave_message}\n\n"
        with open(self.logging_file, 'a') as f:
            f.write(ave_message)

        np.save(os.path.join(self.root_dir, f'losses_{prefix}.npy'), self.test_loss_history)
        
    def log_losses_to_npz(self):
        # Function to reshape loss history
        def reshape_loss(loss_history):
            n_epochs = max(loss_history['epoch']) + 1
            keys = [key for key in loss_history if key != 'epoch']
            reshaped = {}
            for key in keys:
                reshaped[key] = np.array(loss_history[key]).reshape(n_epochs, -1)
            return reshaped
        
        # Reshape train and validation losses
        reshaped_train_loss = reshape_loss(self.train_loss_history)
        reshaped_val_loss = reshape_loss(self.val_loss_history)
        
        # Reshape learning rate history
        reshaped_lr = reshape_loss(self.lr_history)

        np.save(os.path.join(self.root_dir, 'losses_train.npy'), reshaped_train_loss)
        np.save(os.path.join(self.root_dir, 'losses_val.npy'), reshaped_val_loss)
        np.save(os.path.join(self.root_dir, 'lr_history.npy'), reshaped_lr)

    def log_to_console(self, message):
        print(message)
        with open(self.logging_file, 'a') as f:
            f.write(message + '\n')
