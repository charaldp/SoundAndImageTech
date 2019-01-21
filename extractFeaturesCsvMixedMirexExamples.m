features = {'RMS_Energy','Roll_Off_85','Roll_Off_90'};
for i = 1:13
    features{3 + i} = sprintf('MFCC_%d',i);
end
features(17:22) = {'Zero_Crossing_Rate','Spectral_Flatness','Spectral_Kurtosis',...
    'Spectral_Brightness','Spectral_Irregularity','Spectral_Centroid'};
attributes = features;
attributes(23) = {'Class'};

mp3List = dir('MusicSpeech\muspeak-mirex2015-detection-examples\*.mp3');
timestampList = dir('MusicSpeech\muspeak-mirex2015-detection-examples\*.csv');
% !!! Timestamp data have been edited using a text editor's find and
% replace. The speech 's' label has been replaced with 0 and the music 'm'
% label with 1.

% timestampData = csvread('C:\Users\Ðáíáãéþôçò\THMMY\5ï ¸ôïò\9ï ÅîÜìçíï\Ôå÷íïëïãßá ¹÷ïõ êáé Åéêüíáò\Data Sound Files\muspeak-mirex2015-detection-examples\ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv');
% audio=miraudio('C:\Users\Ðáíáãéþôçò\THMMY\5ï ¸ôïò\9ï ÅîÜìçíï\Ôå÷íïëïãßá ¹÷ïõ êáé Åéêüíáò\Data Sound Files\muspeak-mirex2015-detection-examples\ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3');

%timestampData = csvread('C:\Users\Hampo\Desktop\MusicSpeech\muspeak-mirex2015-detection-examples\ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.csv');
%audio=miraudio('C:\Users\Hampo\Desktop\MusicSpeech\muspeak-mirex2015-detection-examples\ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.mp3');

window=0.8;
maxChunk = 200*window;
for j = 1:length(mp3List)
    filename = sprintf('%s\\%s',mp3List(j).folder,mp3List(j).name);
    timestampName = sprintf('%s\\%s',timestampList(j).folder,timestampList(j).name);
    timestampData = csvread(timestampName);
     %% Framed Features
    endSec = timestampData(size(timestampData,1),1) + timestampData(size(timestampData,1),2);
    segments = 0:maxChunk:endSec;
    data = [];
    for l = 1:length(segments)-1
        %% Subsegment Huge files for easier Proccesing
        if l == 1
            audio = miraudio(filename,'Extract',segments(l),segments(l+1),'s','Start');
        elseif l ~= length(segments)-1
            audio = miraudio(filename,'Extract',segments(l)-window/2,segments(l+1),'s','Start');
        else
            audio = miraudio(filename,'Extract',segments(l)-window/2,endSec,'s','Start');
        end
    
        framedAudio = mirframe(audio,'Length',window,'s');
        clear audio;
        framed_rms_energy = mirrms(framedAudio);
        framed_zcr = mirzerocross(framedAudio);

        framedAudioMelSpectrum = mirspectrum(framedAudio,'Mel');
        framed_MFCCs = mirmfcc(framedAudioMelSpectrum);
        clear framedAudioMelSpectrum;
        framedAudioSpectrum = mirspectrum(framedAudio);
        clear framedAudio;
        framed_spectral_roll_off_85 = mirrolloff(framedAudioSpectrum,'Threshold',85);
        framed_spectral_roll_off_90 = mirrolloff(framedAudioSpectrum,'Threshold',90);
        framed_spectral_flatness = mirflatness(framedAudioSpectrum);
        framed_spectral_kurtosis = mirkurtosis(framedAudioSpectrum);
        framed_spectral_brightness = mirbrightness(framedAudioSpectrum);
        framed_spectral_irregularity = mirregularity(framedAudioSpectrum);
        framed_spectral_centroid  = mircentroid(framedAudioSpectrum);
        clear framedAudioSpectrum;

        clear framedAudio;
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
        
        data1 = [raw_rms_energy' raw_spectral_roll_off_85' raw_spectral_roll_off_90'...
            raw_MFCCs' raw_zcr' raw_spectral_flatness' raw_spectral_kurtosis'...
            raw_spectral_brightness' raw_spectral_irregularity'...
            raw_spectral_centroid'];
        data = [data;data1];
    end

    %% Labels
    window_end = (window/2)*(1:size(data,1)) + (window/2);
    window_m = window_end-window/2;
    stamp_end = timestampData(:,1)+timestampData(:,2);
    stamp = [timestampData(:,1) stamp_end timestampData(:,3)];
    a = (window_m > stamp(:,1) & window_m < stamp(:,2));
    label=10*ones(1,size(a,2));
    for i=1:size(a,2)
        if(sum(a(:,i))==1)
            % An instance belongs to only one segment of speech/music
            % of timestampData
            label(i)= timestampData(a(:,i)==1,3);
        else
            % These instances should be removed from the dataset
            label(i)=666;
        end
    end
    cor_windows= label~=666;
    
    data = [data label'];

    [N1,M1] = size(data);

    data = data(cor_windows,:);
    %% Remove Instances that contain NaN values
    sizeInit = size(data,1);
    data = data(~any(isnan(data')),:);
    sizeFin = size(data,1);

    if (sizeInit > sizeFin)
        fprintf('Removed %d instances due to NaN values\n',sizeInit - sizeFin);
    end

    %% Write Data to CSV File
    
    writetable(cell2table([attributes;num2cell(data)]),sprintf('Output/rawDataMirex%s.csv',mp3List(j).name),'writevariablenames',0);
    writetable(cell2table(num2cell(data)),sprintf('Output/noATTRSrawDataMirex%s.csv',mp3List(j).name),'writevariablenames',0);
end
