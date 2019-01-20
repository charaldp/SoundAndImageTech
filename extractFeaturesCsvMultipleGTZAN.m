features = {'RMS_Energy','Roll_Off_85','Roll_Off_90'};
for i = 1:13
    features{3 + i} = sprintf('MFCC_%d',i);
end
features(17:22) = {'Zero_Crossing_Rate','Spectral_Flatness','Spectral_Kurtosis',...
    'Spectral_Brightness','Spectral_Irregularity','Spectral_Centroid'};
attributes = features;
attributes(23) = {'Class'};

window = 0.8; %% seconds

%MusicFileList = dir('C:\Users\Παναγιώτης\THMMY\5ο Έτος\9ο Εξάμηνο\Τεχνολογία Ήχου και Εικόνας\Data Sound Files\music_speech\music_wav\*.wav');
%SpeechFileList = dir('C:\Users\Παναγιώτης\THMMY\5ο Έτος\9ο Εξάμηνο\Τεχνολογία Ήχου και Εικόνας\Data Sound Files\music_speech\speech_wav\*.wav');

MusicFileList = dir('MusicSpeech\GTZAN\music_wav\*.wav');
SpeechFileList = dir('MusicSpeech\GTZAN\speech_wav\*.wav');

%MusicFileList = dir('C:\Users\Hampo\Desktop\MusicSpeech\musan\music\*\*.wav');
%SpeechFileList = dir('C:\Users\Hampo\Desktop\MusicSpeech\musan\speech\*\*.wav');
datasetsInfo={MusicFileList,SpeechFileList};

data = [];
kk = 0;
for i = 1:length(datasetsInfo)  %File Struct Index
    class = mod(i,2);
    for j = 1:length(datasetsInfo{i})                %File Index
        str = sprintf('%s\\%s',datasetsInfo{i}(j).folder,datasetsInfo{i}(j).name);
        audio = miraudio(str);
        framedAudio = mirframe(audio,'Length',window,'s');
        framedAudioSpectrum = mirspectrum(framedAudio,'Window','hamming');
        framedAudioMelSpectrum = mirspectrum(framedAudio,'Mel','Window','hamming');
        %% Framed Features
        framed_rms_energy = mirrms(framedAudio);
        framed_spectral_roll_off_85 = mirrolloff(framedAudioSpectrum,'Threshold',85);
        framed_spectral_roll_off_90 = mirrolloff(framedAudioSpectrum,'Threshold',90);
        framed_MFCCs = mirmfcc(framedAudioMelSpectrum);
        framed_zcr = mirzerocross(framedAudio);
        framed_spectral_flatness = mirflatness(framedAudioSpectrum);
        framed_spectral_kurtosis = mirkurtosis(framedAudioSpectrum);
        framed_spectral_brightness = mirbrightness(framedAudioSpectrum);
        framed_spectral_irregularity = mirregularity(framedAudioSpectrum);
        framed_spectral_centroid  = mircentroid(framedAudioSpectrum);
        
        %% Extract Raw Data
        raw_rms_energy = mirgetdata(framed_rms_energy);
        raw_spectral_roll_off_85 = mirgetdata(framed_spectral_roll_off_85);
        raw_spectral_roll_off_90 = mirgetdata(framed_spectral_roll_off_90);
        raw_MFCCs = mirgetdata(framed_MFCCs);
        raw_zcr = mirgetdata(framed_zcr);
        raw_spectral_flatness = mirgetdata(framed_spectral_flatness);
        raw_spectral_kurtosis = mirgetdata(framed_spectral_kurtosis);
        raw_spectral_brightness = mirgetdata(framed_spectral_brightness);
        raw_spectral_irregularity = mirgetdata(framed_spectral_irregularity);
        raw_spectral_centroid = mirgetdata(framed_spectral_centroid);
        
        raw_class = class * ones(1,length(raw_spectral_irregularity));

        
        dataTemp = [raw_rms_energy' raw_spectral_roll_off_85' raw_spectral_roll_off_90'...
            raw_MFCCs' raw_zcr' raw_spectral_flatness' raw_spectral_kurtosis'...
            raw_spectral_brightness' raw_spectral_irregularity'...
            raw_spectral_centroid' raw_class'];
        sizeInit = size(dataTemp,1);
        dataTemp = dataTemp(~any(isnan(dataTemp')),:);
        sizeFin = size(dataTemp,1);
        %% Remove Instances that contain NaN values
        if (sizeInit > sizeFin)
            fprintf('Removed %d instances due to NaN values\n',sizeInit - sizeFin);
        end
        data = [data;dataTemp];
        clear dataTemp audio framedAudio framedAudioSpectrum framedAudioMelSpectrum...
            framed_rms_energy framed_spectral_roll_off_85 framed_spectral_roll_off_90...
            framed_MFCCs framed_zcr framed_spectral_flatness framed_spectral_kurtosis...
            framed_spectral_brightness framed_spectral_irregularity framed_spectral_centroid...
            raw_rms_energy raw_spectral_roll_off_85 raw_spectral_roll_off_90 raw_MFCCs...
            raw_zcr raw_spectral_flatness raw_spectral_kurtosis raw_spectral_brightness...
            raw_spectral_irregularity raw_spectral_centroid;
        kk = kk + 1;
        fprintf('/==========================================\\\nFeature extraction is currently at file #%d\n\\==========================================/\n',kk);
    end
end

%% Write Data to CSV File

fname=sprintf('Output/rawDataGTZAN.csv');
writetable(cell2table([attributes;num2cell(data)]),fname,'writevariablenames',0);
fname=sprintf('Output/noATTRSrawDataGTZAN.csv');
writetable(cell2table(num2cell(data)),fname,'writevariablenames',0);
