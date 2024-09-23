from utils.keras_utils import train_model, evaluate_model, set_trainable
from models import generate_model

#48: Arousal, 49: Valence, 50: HighLow
DATASET_INDEX = 48

if __name__ == "__main__":
    model = generate_model()

    train_model(model, DATASET_INDEX, dataset_prefix='DEAP_A_c8_b32_OverSample_e2000', epochs=2000, batch_size=32) # epochs 1000 , batch_size=128

    evaluate_model(model, DATASET_INDEX, dataset_prefix='DEAP_A_c8_b32_OverSample_e2000', batch_size=32) # batch_size=128