% All Included scripts must be in the same (root) folder music, files should
% be placed in directories (root)/MusicSpeech. GTZAN dataset subfolders are
% contained in the directory (root)/MusicSpeech/GTZAN/music_wav and
% speech_wav, Musan dataset file are contained in the folders
% (root)/MusicSpeech/musan/music_wav and speech_wav, Mirex Examples 2015
% dataset files should be contained in the folder
% (root)/MusicSpeech/muspeak-mirex2015-detection-examples folder
%% 1. Extract CSV Files

extractFeaturesCsvMultipleGTZAN

extractFeaturesCsvMultipleMusan

extractFeaturesCsvMixedMirexExamples

%% 2. Normalize datasets 

proprocessInstances

%% 3. Classify With Fuzzy Model

trainEvalMultiDataset