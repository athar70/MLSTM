import pickle as cPickle
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import processPPG 

# Constants
Data_path = './Dataset/DEAP/data_preprocessed_python/'

SUBJECT_NUM = 22
VIDEO_NUM = 40

# Input: Valence and arousal values [1..9]
# Output: Encoded label for High or Low Valence-Arousal categories (LL, HH, HL, LH) mapped to [1..4]
def encodeValenceArousalTOHighLow(valence, arousal):
    if valence < 5 and arousal < 5:
        HL_cat = 'LL'
    elif valence >= 5 and arousal >= 5:
        HL_cat = 'HH'
    elif valence >= 5 and arousal < 5:
        HL_cat = 'HL'
    elif valence < 5 and arousal >= 5:
        HL_cat = 'LH'

    label_HL = 1 if HL_cat == 'LL' else 2 if HL_cat == 'HH' else 3 if HL_cat == 'HL' else 4 if HL_cat == 'LH' else 0
    return label_HL


# Oversampling function to handle class imbalance
def overSampleData(segments, labels):
    """
    Input: 3D segments and corresponding labels
    Output: Oversampled data and labels
    """
    ros = RandomOverSampler(random_state=42)
    ros.fit_resample(segments[:, 0, :], labels)  # Oversample based on the first feature dimension
    indices = ros.sample_indices_

    segments_oversample = segments[indices, :, :]
    labels_oversample = labels[indices]
    
    return segments_oversample, labels_oversample

# Function to visualize class distribution as a histogram
def classDistributionHist(labels, title=''):
    """
    Input: Labels and a title string
    Output: Plot displaying the class distribution
    """
    range_labels = np.unique(labels).size
    arr = plt.hist(labels, bins=np.arange(range_labels + 1 + 1) - 0.5, ec="k")
    xint = range(1, range_labels + 1)
    plt.xticks(xint)
    plt.title(f'{title} class distribution')
    plt.xlabel('Class Number')
    plt.ylabel('Occurrences')

    for i in xint:
        occurrences = np.count_nonzero(labels == i)
        per = (occurrences / labels.shape[0]) * 100
        plt.text(arr[1][i], arr[0][i], f'{per:.1f}%')

    plt.show()

# Normalize the segments using z-score normalization
def normalizeSegments(segments):
    """
    Input: 3D segment data (samples, features, time steps)
    Output: Normalized segment data
    """
    shape = segments.shape
    scaler = MinMaxScaler()
    All_segments = np.transpose(segments, (0, 2, 1)).reshape(shape[0] * shape[2], shape[1])

    segments_normalize = scaler.fit_transform(All_segments)
    segments_normalize = segments_normalize.reshape(shape[0], shape[2], shape[1])
    segments_normalize = np.transpose(segments_normalize, (0, 2, 1))
    return segments_normalize

# Function to split the data into training and testing sets
def splitTrainTest(segments, labels, test_size=0.3, RANDOM_SEED=42):
    """
    Input:
    - segments: The dataset, typically 3D (samples, features, time steps)
    - labels: The corresponding labels for the dataset
    - test_size: The proportion of the dataset to include in the test split (default: 30%)
    - RANDOM_SEED: Seed for random number generation (ensures reproducibility)

    Output:
    - X_train: Training data
    - X_test: Testing data
    - y_train: Training labels
    - y_test: Testing labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        segments, labels, test_size=test_size, random_state=RANDOM_SEED
    )
    
    return X_train, X_test, y_train, y_test


# Save train and test data to specified directory
def saveDataTrainTest(X_train, y_train, X_test, y_test, category):
    """
    Input: Train and test data, category string
        category name: Arousal, Valence, HighLow 
    Saves the data to data folder
    """
    Save_Data_path = f'./data/DEAP/{category}'
    
    np.save(os.path.join(Save_Data_path, 'y_test.npy'), y_test)
    np.save(os.path.join(Save_Data_path, 'y_train.npy'), y_train)
    np.save(os.path.join(Save_Data_path, 'X_test.npy'), X_test)
    np.save(os.path.join(Save_Data_path, 'X_train.npy'), X_train)

# Split, oversample, normalize, and save the data
def processTrainTest(segments, labels, category):
    """
    Input: Segments, labels, and category string (Arousal, Valence, HighLow)
    This function normalizes the data, splits it into train/test sets,
    applies oversampling, and saves the processed data.
    """
    segments_normalize = normalizeSegments(segments)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = splitTrainTest(segments_normalize, labels)

    # Apply oversampling to balance the classes
    X_train, y_train = overSampleData(X_train, y_train)
    X_test, y_test = overSampleData(X_test, y_test)

    print("Train: ", X_train.shape, " , Test: ", X_test.shape)

    # Save the processed data
    saveDataTrainTest(X_train, y_train, X_test, y_test, category)


if __name__ == "__main__":
    segments = []
    labels_A = []
    labels_V = []
    labels_HL = []

    # Loop over each subject and process the data
    for s in range(1, SUBJECT_NUM + 1):
        file_name = f"s{str(s).zfill(2)}.dat"
        dat_file = os.path.join(Data_path, file_name)

        # Load the data file for each subject
        with open(dat_file, 'rb') as f:
            x = cPickle.load(f, encoding='latin1')

        data = x['data']  # 40 x 40 x 8064 : video/trial x channel x data
        labels = x['labels']  # 40 x 4 : video/trial x label (valence, arousal, dominance, liking)

        ## each array would contain 5 seconds of data and the next array would overlap the previous by 2 seconds. (128 Hz)
        '''
        37-1	GSR (values from Twente converted to Geneva format (Ohm))
        38-1	Respiration belt
        39-1	Plethysmograph
        40-1	Temperature
        '''
        # Process each video for the subject
        for v in range(VIDEO_NUM):
            video_data = data[v, 36:40, :]  # Select specific channels: GSR, Respiration, Plethysmograph, Temperature
            video_label = np.round(labels[v, :]).astype(int)  # Convert labels to integers

            SAMPLE_RATE = 128  # Hz
            N_TIME_STEPS = 5 * SAMPLE_RATE  # 5 seconds
            step = 3 * SAMPLE_RATE  # 3 seconds overlap

            # Simulate HRV (Heart Rate Variability) and AVG_HRV (Average HRV) calculation
            HRV, AVG_HRV = processPPG.ProcessSignalPPG(video_data[2])

            # Segment data with 5-second windows and 2-second overlaps
            for i in range(0, video_data.shape[1] - N_TIME_STEPS, step):
                video_GSR = video_data[0, i: i + N_TIME_STEPS]
                video_RES = video_data[1, i: i + N_TIME_STEPS]
                video_PPG = video_data[2, i: i + N_TIME_STEPS]
                video_TEMP = video_data[3, i: i + N_TIME_STEPS]
                video_HRV = HRV[i: i + N_TIME_STEPS]

                video_valence = video_label[0]
                video_arousal = video_label[1]
                video_label_HL = encodeValenceArousalTOHighLow(video_valence, video_arousal)

                # Store segmented data and labels
                segments.append([video_GSR, video_RES, video_PPG, video_TEMP, video_HRV])
                labels_V.append([video_valence])
                labels_A.append([video_arousal])
                labels_HL.append([video_label_HL])

    # Convert lists to numpy arrays
    segments = np.array(segments)
    labels_A = np.array(labels_A)
    labels_V = np.array(labels_V)
    labels_HL = np.array(labels_HL)

    print(segments.shape)
    print(labels_A.shape)

    # Example usage for Arousal category
    processTrainTest(segments, labels_A, 'Arousal')
    # Uncomment for other categories
    # processTrainTest(segments, labels_V, 'Valence')
    # processTrainTest(segments, labels_HL, 'HighLow')
