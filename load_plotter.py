import matplotlib.pyplot as plt

class LossPlotter:
    def __init__(self, history):
        """
        history: object trả về từ model.fit()
        """
        self.history = history

    def plot(self, save_path='train_val_loss.png', figsize=(8,6), dpi=300):
        train_loss = self.history.history.get('loss')
        val_loss = self.history.history.get('val_loss')
        if train_loss is None or val_loss is None:
            print("Không tìm thấy 'loss' hoặc 'val_loss' trong history!")
            return

        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_loss, label='Train Loss', color='blue')
        plt.plot(epochs, val_loss, label='Validation Loss', color='red')
        plt.title('Train vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        print(f"Diagram saved in file: {save_path}")
