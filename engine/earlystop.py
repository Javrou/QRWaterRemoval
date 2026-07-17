class EarlyStopping:

    def __init__(
            self,
            patience,
            min_epochs,
            target_zxing,
            min_delta_loss=1e-3,
            min_delta_zxing=1e-3
    ):

        self.patience = patience
        self.min_epochs = min_epochs
        self.target_zxing = target_zxing

        self.min_delta_loss = min_delta_loss
        self.min_delta_zxing = min_delta_zxing

        self.counter = 0

    def step(
            self,
            epoch,
            loss_improved,
            zxing_improved,
            best_zxing
    ):

        if best_zxing < self.target_zxing:
            self.counter = 0
            return False

        if loss_improved or zxing_improved:
            self.counter = 0
        else:
            self.counter += 1

        return (
                epoch >= self.min_epochs
                and
                self.counter >= self.patience
        )