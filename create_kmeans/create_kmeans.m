%% file to compute for each trial the gini value in a window of 100 ms
clear all; % close all;

addpath('/home/paolo/cvsa/ic_cvsa_ws/src/analysis_cvsa/EOG')
addpath('/home/paolo/cvsa/ic_cvsa_ws/src/analysis_cvsa/equal_ros')

%% Initialization
bands = [{[6 11]} {[8 14]} {[11 16]}];
bands_str = cellfun(@(x) sprintf('%d-%d', x(1), x(2)), bands, 'UniformOutput', false);
DATAPAH = '/home/paolo/cvsa/ic_cvsa_ws/src/kmeans_cvsa/cfg/';
nbands = length(bands);
signals = cell(1, nbands);
headers = cell(1, nbands);
for idx_band = 1:nbands
    headers{idx_band}.TYP = [];
    headers{idx_band}.DUR = [];
    headers{idx_band}.POS = [];
    signals{idx_band} = [];
end
classes = [730 731];      
cf_event = 781;
fix_event = 786;
nchannels = 39;
nclasses = length(classes);
filterOrder = 4;
avg = 1;% 0.75;
eog_threshold = 500;

%% Load file
[filenames, pathname] = uigetfile('*.gdf', 'Select GDF Files', 'MultiSelect', 'on');
if ischar(filenames)
    filenames = {filenames};
end
subject = filenames{1}(1:2);

th_high = inf;

%% concatenate the files
nFiles = length(filenames);
trial_with_eog = [];
trial_with_high_voltage = [];
for idx_file= 1: nFiles
    fullpath_file_shift = fullfile(pathname, filenames{idx_file});
    disp(['file (' num2str(idx_file) '/' num2str(nFiles)  '): ', filenames{idx_file}]);
    [c_signal,header] = sload(fullpath_file_shift);
    c_signal = c_signal(:,1:nchannels);
    channels_label = header.Label;
    sampleRate = header.SampleRate;

    c_trial_with_eog = eog_detection(c_signal, header, eog_threshold, {'FP1', 'FP2', 'EOG'});
    trial_with_eog = [trial_with_eog; c_trial_with_eog];

    disp('   [proc] power band');
    for idx_band = 1:nbands
        band = bands{idx_band};

        % for power band using hilbert transformation
        bufferSize = floor(avg*sampleRate);
        chunkSize = 32;
        [signal_processed, header_processed] = processing_onlineROS_hilbert(c_signal, header, nchannels, bufferSize, filterOrder, band, chunkSize);

        if all(subject == 'h8')
            signal_processed(:,23) = 0;
        end

        if all(band == [8 14])
            c_trial_with_high_voltage = check_voltage_trial(signal_processed, header_processed, th_high);
            trial_with_high_voltage = [trial_with_high_voltage; c_trial_with_high_voltage];
        end

        c_header = headers{1, idx_band};
        c_header.sampleRate = header_processed.SampleRate/chunkSize;
        c_header.channels_labels = header_processed.Label;
        if isempty(find(header_processed.EVENT.TYP == 2, 1)) % no eye calibration
            c_header.TYP = cat(1, c_header.TYP, header_processed.EVENT.TYP);
            c_header.DUR = cat(1, c_header.DUR, header_processed.EVENT.DUR);
            c_header.POS = cat(1, c_header.POS, header_processed.EVENT.POS + size(signals{1, idx_band}, 1));
        else
            k = find(header_processed.EVENT.TYP == 1, 1);
            c_header.TYP = cat(1, c_header.TYP, header_processed.EVENT.TYP(k:end));
            c_header.DUR = cat(1, c_header.DUR, header_processed.EVENT.DUR(k:end));
            c_header.POS = cat(1, c_header.POS, header_processed.EVENT.POS(k:end) + size(signals{1, idx_band}, 1));
        end
        signals{1, idx_band} = cat(1, signals{1, idx_band}, signal_processed(:,:));
        headers{1, idx_band} = c_header;
    end
end


%% Labelling data 
events = headers{1,1};
sampleRate = events.sampleRate;
cuePOS = events.POS(ismember(events.TYP, classes));
cueDUR = events.DUR(ismember(events.TYP, classes));
cueTYP = events.TYP(ismember(events.TYP, classes));

fixPOS = events.POS(events.TYP == 786);
fixDUR = events.DUR(events.TYP == 786);

cfPOS = events.POS(events.TYP == 781);
cfDUR = events.DUR(events.TYP == 781);

minDurCue = min(cueDUR);
minDurFix = min(fixDUR);
ntrial = length(cuePOS);

%% Labeling data for the dataset
min_durFIX = min(fixDUR);
min_durCF = min(cfDUR);
min_durCUE = min(cueDUR);

trial_start = nan(ntrial, 1);
trial_end = nan(ntrial, 1);
trial_typ = nan(ntrial, 1);
for idx_trial = 1:ntrial
    trial_start(idx_trial) = fixPOS(idx_trial);
    trial_typ(idx_trial) = cueTYP(idx_trial);
    trial_end(idx_trial) = cfPOS(idx_trial) + cfDUR(idx_trial) - 1;
end

min_trial_data = min(trial_end - trial_start+1);
trial_data = nan(min_trial_data, nbands, nchannels, ntrial);
for idx_band = 1:nbands
    c_signal = signals{idx_band};
    for trial = 1:ntrial
        c_start = trial_start(trial);
        c_end = trial_start(trial) + min_trial_data - 1;
        trial_data(:,idx_band,:,trial) = c_signal(c_start:c_end,:);
    end
end

%% refactoring the data
% now the data are placed alternatevely, so odd trial id is for 730, even for 731
trial_with_eog = trial_with_high_voltage | trial_with_eog;
balanced_trial_idx = balanced_data_afterEOG(trial_with_eog, trial_typ, classes);
balanced_trial_data = trial_data(:,:,:,logical(balanced_trial_idx));
trial_typ = trial_typ(logical(balanced_trial_idx));
ntrial = sum(balanced_trial_idx);
idx_classes_trial = nan(ntrial/2, nclasses);
for idx_class = 1:nclasses
    idx_classes_trial(:,idx_class) = find(trial_typ == classes(idx_class));
end

tmp = nan(size(balanced_trial_data));
trial_typ = nan(size(trial_typ));
i = 1;
for idx_trial_class = 1:2:ntrial
    for idx_class = 1:nclasses
        tmp(:,:,:,idx_trial_class + idx_class - 1) = balanced_trial_data(:,:,:,idx_classes_trial(i, idx_class));
        trial_typ(idx_trial_class + idx_class - 1) = classes(idx_class);
    end
    i = i + 1;
end
trial_data = tmp; % samples x bands x channels x trials
trial_data(:,:,[1, 2, 19],:) = 0; % remove the power of the EOG channel, FP1 anf FP2 --> also in sparsity

%% compute sparsity
% define regions
occipital = {'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'OZ'}; [~, ch_occipital] = ismember(occipital, channels_label);
central = {'CP3', 'CP1', 'C3', 'FC3', 'C1', 'FC1', 'CP4', 'C4', 'FC4', 'CP2', 'C2', 'FC2', 'FCZ', 'CZ'}; [~, ch_central] = ismember(central, channels_label);
frontal = {'F3', 'F1', 'F2', 'F4', 'FP1', 'FP2', 'FZ'}; [~, ch_frontal] = ismember(frontal, channels_label);
nchannelsSelected = size(ch_occipital, 2);

nsparsity = 3;
sparsity = nan(min_trial_data, nbands, ntrial, nsparsity); % sample x band x trial x sparsity

for c = 1:ntrial
    c_data = squeeze(trial_data(:,:,:,c)); % samples x band x channels

    for sample = 1:min_trial_data
        c_sample = squeeze(c_data(sample,:,:)); % bands x channels

        for idx_band = 1:nbands
            tmp = squeeze(c_sample(idx_band,:)); % 1 x channels

            [sparsity(sample, idx_band, c,:), label_sparsity, o_l, o_r, frontal, c_l, c_r, excluded_chs] = compute_features_kmeans(tmp);
        end
    end
end

%% ----------------- KMEANS -----------------
K = 2;
choosen_band = 2; % choose 8-14

sparsity_cf = squeeze(sparsity(minDurFix+minDurCue+1:end, choosen_band,:,:));

% z-score -> train and use the mu and var also for the test
data_3D = sparsity_cf(:, :, :);
data_2D = reshape(data_3D, size(data_3D, 1) * size(data_3D,2), size(data_3D,3));
mu_features = mean(data_2D, 1);
sigma_features = std(data_2D, 0, 1);
sigma_features(sigma_features == 0) = eps;
data_standardized_2D = (data_2D - mu_features) ./ sigma_features;
data_standardized_3D = reshape(data_standardized_2D, size(data_3D, 1), ntrial, size(data_3D,3));

disp('Kmeans execution...');

[idx_train, C] = kmeans(data_standardized_2D, K, ...
    'Distance', 'sqeuclidean', ...
    'Replicates', 10, ...  % 'Replicates' è fondamentale per un risultato robusto
    'Display', 'final');

[~, sortIdx] = sort(C(:, 2), 'descend'); % Ordina in base alla feature 2 (OFI)
C = C(sortIdx, :);

% compute the labels for all the trials, with the selected centroids
train_kmeans = nan(ntrial * size(sparsity_cf, 1), nsparsity);
for c = 1:ntrial
    train_kmeans((c-1)*size(sparsity_cf, 1) + 1: c * size(sparsity_cf, 1),:) = sparsity_cf(:,c,:);
end
distances = pdist2(train_kmeans, C);  
[~, raw_labels] = min(distances, [], 2);

% save the cluster for each data
cluster_labels_train = nan(size(sparsity_cf, 1), ntrial);
for c = 1:ntrial
    cluster_labels_train(:,c) = raw_labels((c-1)*size(sparsity_cf, 1) + 1: c * size(sparsity_cf, 1));
end

%% save the kmeans
save_kmeans(C, mu_features, sigma_features, filenames, DATAPAH, subject, o_l, o_r, frontal, c_l, c_r, excluded_chs, channels_label, bands(choosen_band))

%% extract and save data for the QDA


%% ----------- FUNCTIONS --------
% save kmeans
function save_kmeans(C, mu_features, sigma_features, files, DATAPAH, subject, o_l_idx, o_r_idx, frontal_idx, c_l_idx, c_r_idx, excluded_chs, channels_labels, band)
    % --- Dati K-Means ---
    % C:               centroids 
    % mu_features:     mean of the data
    % sigma_features:  std of the data
    % files:           file from which the kmeans is trained
    % DATAPAH:         where to save
    % subject:         subject of the experiment

    % mu and sigma as string 
    muStr = strjoin(arrayfun(@(x) sprintf('%.8f', x), mu_features, 'UniformOutput', false), ', ');
    sigmaStr = strjoin(arrayfun(@(x) sprintf('%.8f', x), sigma_features, 'UniformOutput', false), ', ');

    % centroids as a string matrix
    K = size(C,1);
    rowStrings = cell(K, 1);
    for i = 1:K
        c_row = C(i, :); 
        innerListStr = strjoin(arrayfun(@(x) sprintf('%.8f', x), c_row, 'UniformOutput', false), ', ');
        rowStrings{i} = sprintf('    - [%s]', innerListStr);
    end
    centroidsStr = strjoin(rowStrings, '\n');

    filenamesStr = strjoin(files, ';\n');

    rowStrings = cell(size(band,2), 1);
    for i = 1:K
        c_row = band{i}; 
        innerListStr = strjoin(arrayfun(@(x) sprintf('%d', x), c_row, 'UniformOutput', false), ', ');
        rowStrings{i} = sprintf('    - [%s]', innerListStr);
    end
    bands_str = strjoin(rowStrings, '\n');

    o_l_channels = string(channels_labels(o_l_idx));
    o_l_channels = "'" + o_l_channels + "'";
    o_l_channels = join(o_l_channels, ", ");
    o_l_str = strjoin(arrayfun(@(x) sprintf('%d', x), o_l_idx, 'UniformOutput', false), ', ');

    o_r_channels = string(channels_labels(o_r_idx));
    o_r_channels = "'" + o_r_channels + "'";
    o_r_channels = join(o_r_channels, ", ");
    o_r_str = strjoin(arrayfun(@(x) sprintf('%d', x), o_r_idx, 'UniformOutput', false), ', ');

    frontal_channels = string(channels_labels(frontal_idx));
    frontal_channels = "'" + frontal_channels + "'";
    frontal_channels = join(frontal_channels, ", ");
    frontal_str = strjoin(arrayfun(@(x) sprintf('%d', x), frontal_idx, 'UniformOutput', false), ', ');

    c_l_channels = string(channels_labels(c_l_idx));
    c_l_channels = "'" + c_l_channels + "'";
    c_l_channels = join(c_l_channels, ", ");
    c_l_str = strjoin(arrayfun(@(x) sprintf('%d', x), c_l_idx, 'UniformOutput', false), ', ');

    c_r_channels = string(channels_labels(c_r_idx));
    c_r_channels = "'" + c_r_channels + "'";
    c_r_channels = join(c_r_channels, ", ");
    c_r_str = strjoin(arrayfun(@(x) sprintf('%d', x), c_r_idx, 'UniformOutput', false), ', ');

    excl_channels = string(channels_labels(excluded_chs));
    excl_channels = "'" + excl_channels + "'";
    excl_channels = join(excl_channels, ", ");
    excl_str = strjoin(arrayfun(@(x) sprintf('%d', x), excluded_chs, 'UniformOutput', false), ', ');

    % build the yaml
    yamlContent = sprintf(['KmeansModelCfg:\n' ...
                     '  name: "kmeans_model"\n' ...
                     '  filenames: "%s"\n'...
                     '  params:\n' ...
                     '    K: %d\n' ...
                     '    nfeatures: %d\n' ...
                     '    occipital_left_idx: [%s]\n' ...
                     '    occipital_left: [%s]\n' ...
                     '    occipital_right_idx: [%s]\n' ...
                     '    occipital_right: [%s]\n' ...
                     '    frontal_idx: [%s]\n' ...
                     '    frontal: [%s]\n' ...
                     '    central_left_idx: [%s]\n' ...
                     '    central_left: [%s]\n' ...
                     '    central_right_idx: [%s]\n' ...
                     '    central_right: [%s]\n' ...
                     '    excluded_idx: [%s]\n' ...
                     '    excluded: [%s]\n' ...
                     '    mu: [%s]\n' ...
                     '    sigma: [%s]\n' ...
                     '    centroids: \n%s\n' ...
                     '    band: \n%s\n'], ...
                     filenamesStr, ...
                     K, ...
                     size(C,2), ...
                     o_l_str, ...
                     o_l_channels, ...
                     o_r_str, ...
                     o_r_channels, ...
                     frontal_str, ...
                     frontal_channels, ...
                     c_l_str, ...
                     c_l_channels, ...
                     c_r_str, ...
                     c_r_channels, ...
                     excl_str, ...
                     excl_channels, ...
                     muStr, ...
                     sigmaStr, ...
                     centroidsStr,...
                     bands_str);

    file_name = [DATAPAH 'kmeans_' subject '.yaml'];
    fileID = fopen(file_name, 'w');
    fprintf(fileID, '%s', yamlContent);
    fclose(fileID);

    disp(['Modello K-Means salvato in ', file_name]);
end
% sparsity 
function [sparsity, label_sparsity, o_l, o_r, frontal, c_l, c_r, excluded_chs] = compute_features_kmeans(c_signal)
    % take the one which contribute at the 95% of the energy
    sparsity = nan(3,1);
    label_sparsity = [{'LI'},{'GI'},{'GB'}];
    o_l = sort([29 13 30 37 33 34 17]); o_r = sort([31 15 32 35 36 38 18]);
    frontal = sort([3 4 5 20 21]); c_l = sort([6 22 25 8 11 27]); c_r = sort([7 24 26 10 12 28]);
    excluded_chs = [1,2,19];

    % --- LAP --- Calcola il LAP per tutti i punti nella finestra passata (window_signal)
    % show the occipital lateralization that is strongand present during the CVSA
    P_left_window  = mean(c_signal(o_l));
    P_right_window = mean(c_signal(o_r));
    LAP_history = (P_right_window - P_left_window) ./ (P_right_window + P_left_window + eps);
    sparsity(1) = abs(LAP_history); % LAP_Mean

    % --- Gini Index + Occipital Power ---  -> when high there is a zone stronger, so IC
    % show the focusing is weighted in with the power in the occipital part, in this way ig CVSA then strong value 
    non_zeros_chs = setdiff(1:size(c_signal,2), excluded_chs);
    global_mean = mean(c_signal(non_zeros_chs)); % car filter
    current_signal_normalized = c_signal - global_mean; % remove the global energy
    mean_roi = [mean(current_signal_normalized(frontal)), mean(current_signal_normalized(c_l)), ...
        mean(current_signal_normalized(c_r)), mean(current_signal_normalized(o_l)), ...
        mean(current_signal_normalized(o_r))];
    mean_roi = abs(mean_roi); % make sure the energy is positive--> we are using peak and valli with same significance
    mean_roi_ordered = sort(mean_roi);
    n = length(mean_roi_ordered);
    sum_roi_p = 0;
    for i = 1:n
        sum_roi_p = sum_roi_p + (n+1-i) * mean_roi_ordered(i);
    end
    total_sum = sum(mean_roi_ordered);
    if total_sum > 0
        gi = (1/n) * (n+1-2*sum_roi_p/total_sum);
    else
        gi = 0;
    end
    % compute the weight factor
    pot_occipital = mean_roi(4) + mean_roi(5);
    pot_total_roi = sum(mean_roi); % Somma di F, CL, CR, OL, OR
    if pot_total_roi > 0
        occipital_power = pot_occipital / pot_total_roi;
    else
        occipital_power = 0; 
    end
    sparsity(2) = occipital_power * gi;

    % --- GB ---
    % return the global power mean, 
    sparsity(3) = global_mean;
end