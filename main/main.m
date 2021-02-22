%% Example Of Classification Using Image Data
%
% Image classification involves determining if an image contains some 
% specific object, feature, or activity.
% The goal of this particular example is to provide a strategy to
% construct a classifier that can automatically detect which animal we are
% looking at.
%
% - input data: imageDatastore
% - detection of the characteristic features on the images (bag of
%   features)
% - training classifier (KRLS)
% - analysing results
%
% Dependencies:
% > readAndResizeImages.m
% > normalize.m 
% > selectModel_KRLS_CompleteCrossValidation.m
% > trainModel_KRLS_MulticlassClassification.m
% > applyModel_KRLS_MulticlassClassification.m
%
% The functions were introduced by Luca Oneto during the MLDA course.
%
% Created: R2020b, January 2021, Patryk Jan Sozański

clear
close all;
clc

%% Description Of The Data
%
% The dataset contains images of 3 different types of animals: dogs, cats
% and wild animals.
% The images are photos of the animals in various colour and shape
% variations that have been taken from different angles, positions, and
% different lighting conditions. 
% These variations make this a challenging task.
%
% The data set has been downloaded from www.kaggle.com.

%% Loading Image Data
%
% Creating Image Datastore Object from local directory.
% The directory "animals" contains subdirectories, one for each animal
% type.

imds = imageDatastore('../dane/animals', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Reading and resizing images to one common format.
imds.ReadFcn = @readAndResizeImages;

%% Exploratory Data Analysis
%
% Investigating the dataset to summarize its main characteristics and
% determine how best to manipulate the data source to get the answer.

% Displaying class names and counts.
countCategories = countEachLabel(imds);
categories = countCategories.Label;

% Displaying samples of image data.
sample = splitEachLabel(imds, 16, 'randomize');

for ii = 1:length(categories)
    montage(sample.Files((16 * (ii - 1) + 1):(16 * (ii - 1) + 16)));
    title(char(countCategories.Label(ii)));
    pause(2);
    close all
end

clear ii;
clear sample;

%% Pre-processing Of The Training Data
%
% Encoding and transforming the data to bring it to such a state that it
% could be easily parsed by the algorithm.

%% Data Shuffling
%
% Shuffling dataset to avoid any element of bias/patterns in the split
% datasets before training the model and to provide identical probability
% distribution.

imds = shuffle(imds);

%% Partitioning Images Into Training And Validation Set And Test Set
%
% Training set for fitting the model's parameters.
% Validation set for providing an unbiased evaluation of the model fit on
% the training set while tuning the model's hyperparameters.
% Test set for providing an unbiased evaluation of the model fit on the
% training and validation set.

%[training_set, test_set] = imds.splitEachLabel(300, 50, 'randomize', true);

[training_set, test_set] = splitEachLabel(imds, 0.7);

%% Feature Extraction Using Bag Of Features
%
% Bag of features, also known as bag of visual words is one way to detect
% and extract features from images. To represent an image using this
% approach, an image can be treated as a document and occurance of visual
% "words" in images are used to generate a histogram that represents an
% image.
%
% The bag of features provides the data that has no missing values and no
% categorical variables.

%% Creating Visual Vocabulary
%
% Extraction of vectors of features from images.
% Each image described by the same amout of the visual words and their
% occurances.

tic
bag = bagOfFeatures(training_set, ...
    'VocabularySize', 250, ...
    'PointSelection', 'Detector');
animalData = double(encode(bag, training_set));
toc

%% Creating The Array Of Responses For The Training Set
%
% Encoding the categories of the animals into simple numerical code that
% can be used by the algorithm.

animalType = [];
for i = 1:size(animalData, 1)
    if strcmp(string(training_set.Labels(i)), 'cat')
        animalType = [animalType; 1]; %#ok<AGROW>
    elseif strcmp(string(training_set.Labels(i)), 'dog')
        animalType = [animalType; 2]; %#ok<AGROW>
    else
        animalType = [animalType; 3]; %#ok<AGROW>
    end
end

clear i;

%% Data Normalization
%
% Changing the values of numeric columns to a common scale.
% 
% (The data extracted from images using Bag of Features does not have to be
% normalized.)

animalData = normalize(animalData);

%% Visualization Of Feature Vectors
%
% Visualization of the feature vector representations of a random image
% from each category.

imgs = training_set.splitEachLabel(1, 'randomize', true);

img = readimage(imgs, 1);
featureVector = encode(bag, img);
subplot(3, 2, 1);
imshow(img);
subplot(3, 2, 2);
bar(featureVector);
title('Visual Word Occurrences');
xlabel('Visual Word Index');
ylabel('Frequency');

img = readimage(imgs, 2);
featureVector = encode(bag, img);
subplot(3, 2, 3);
imshow(img);
subplot(3, 2, 4); 
bar(featureVector);
title('Visual Word Occurrences');
xlabel('Visual Word Index');
ylabel('Frequency');

img = readimage(imgs, 3);
featureVector = encode(bag, img);
subplot(3, 2, 5);
imshow(img);
subplot(3, 2, 6); 
bar(featureVector);
title('Visual Word Occurrences');
xlabel('Visual Word Index');
ylabel('Frequency');

pause(5);
close all;

clear featureVector;
clear imgs;
clear img;

%% Model Selection
%
% Selecting one final machine learning model from among a collection of
% candidate machine learning models for a training dataset.
%
% In this example it is applied across Kernel-Based Regularized Least
% Squares models configured with different model hyperparameters (lambda,
% gamma).

tic
lambda = logspace(-3, 2, 8);
gamma = logspace(-3, 2, 8);
[gamma, lambda] = selectModel_KRLS_CompleteCrossValidation(animalData, animalType, lambda, gamma, true); 
toc

%% Final Model Creation
%
% Creation of the model with the best hyperparameters chosen in MS phase.

model = trainModel_KRLS_MulticlassClassification(animalData, animalType, lambda, gamma);

%% Preparation Of The Test Set
%
% Encoding data from the bag of features for the test set.
% Normalisation of the test set.
% Creation of the array of responses for the test set.

testAnimalData = double(encode(bag, test_set));
testAnimalData = normalize(testAnimalData);

actualAnimalType = [];
for i = 1:size(testAnimalData, 1)
    if strcmp(string(test_set.Labels(i)), 'cat')
        actualAnimalType = [actualAnimalType; 1]; %#ok<AGROW>
    elseif strcmp(string(test_set.Labels(i)), 'dog')
        actualAnimalType = [actualAnimalType; 2]; %#ok<AGROW>
    else
        actualAnimalType = [actualAnimalType; 3]; %#ok<AGROW>
    end
end

%% Testing Out The Accuracy Of The Final Model On The Test Set
%
% Applying the final model on the test set.
% Calculating the accuracy of the final model.

predictedOutcome = applyModel_KRLS_MulticlassClassification(model, testAnimalData);
correctPredictions = (predictedOutcome == actualAnimalType);
validationAccuracy = mean(correctPredictions);

clear i;

%% Testing Performance Of The Final Model
%
% Visualization of the confusion matrix.

fprintf('\n');
fprintf('Validation Accuracy: %.2f%%\n', round(validationAccuracy * 100, 2));
fprintf('\n');
for i = 1:3
    if i == 1
        fprintf('cat \t');
    elseif i == 2
        fprintf('dog \t');
    else
        fprintf('wild \t');
    end
    for j = 1:3
        fprintf('%d \t',sum(predictedOutcome(actualAnimalType == i) == j));
    end
    fprintf('\n');
end
fprintf('T/P\tcat \tdog \twild\n');

clear i;
clear j;

%% Visualizing Examples Of The Classifier's Predictions

for i = 1:6
    figure(2);
    randomNumber = randi(length(test_set.Labels));
    
    img = test_set.readimage(randomNumber);
    
    imshow(img);
    
    bestGuess = predictedOutcome(randomNumber);

    if bestGuess == 1
        bestGuess = 'cat';
    elseif bestGuess == 2
        bestGuess = 'dog';
    else
        bestGuess = 'wild';
    end

    if bestGuess == test_set.Labels(randomNumber)
        titleColor = 'g';
    else
        titleColor = 'r';
    end
    title(sprintf('Best Guess: %s\nActual: %s',...
        char(bestGuess),test_set.Labels(randomNumber)),...
        'color', titleColor)
    pause(3);
    close all;
end

clear i;
clear bestGuess;
clear img;
clear randomNumber;
clear titleColor;

%% Visualizing Examples Of The Classifier's Mistakes

j = 1;
for i = randi(length(actualAnimalType)):length(actualAnimalType)
    
    if i == length(actualAnimalType)
            pause(5);
            close all;
            break;
    end
    
    if (actualAnimalType(i) ~= predictedOutcome(i))
        
        if j < 10
            
            figure(3)
            img = test_set.readimage(i);
            subplot(3, 3, j);
            imshow(img);

            bestGuess = predictedOutcome(i);
            trueType = actualAnimalType(i);

            if bestGuess == 1
                bestGuess = 'cat';
            elseif bestGuess == 2
                bestGuess = 'dog';
            else
                bestGuess = 'wild';
            end

            if trueType == 1
                trueType = 'cat';
            elseif trueType == 2
                trueType = 'dog';
            else
                trueType = 'wild';
            end

            title(sprintf('Best Guess: %s\nActual: %s',bestGuess, trueType), 'color', 'r');
            
            j = j + 1;
            
        else
            pause(5);
            close all;
            break;
        end
    end
end

clear i;
clear j;
clear bestGuess;
clear trueType;
clear img;