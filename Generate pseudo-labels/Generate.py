import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('TkAgg')


def beat_template(ecg_signal):
    lead_labels=[]
    for lead in range(12):
        real_r_peaks=[]
        up_peaks, _=find_peaks(ecg_signal[lead], distance=375)
        up_peaks=up_peaks.tolist()
        inverted_signal = [-x for x in ecg_signal[lead]]
        inverted_peaks, _ = find_peaks(inverted_signal, distance=375)
        inverted_peaks = inverted_peaks.tolist()

        min_length = min(len(up_peaks), len(inverted_peaks))

        abs_count1 = sum(1 for i in range(min_length) if abs(ecg_signal[lead][up_peaks[i]]) > abs(ecg_signal[lead][inverted_peaks[i]]))
        abs_count2 = sum(1 for i in range(min_length) if abs(ecg_signal[lead][inverted_peaks[i]]) > abs(ecg_signal[lead][up_peaks[i]]))

        if abs_count1 > abs_count2:
            real_r_peaks=up_peaks
        elif abs_count1 < abs_count2:
            real_r_peaks=inverted_peaks
        elif abs_count1 == abs_count2:
            real_r_peaks=up_peaks


        window_size = 400

        # 存储截取的心拍数据
        heartbeats = []
        rr_intervals = []


        X=0

        # Traverse the R-peak position, intercept each heart beat and calculate the RR interval
        for i in range(1, len(real_r_peaks)):
            r_peak = real_r_peaks[i]
            previous_r_peak = real_r_peaks[i - 1]
            rr_interval = r_peak - previous_r_peak
            rr_intervals.append(rr_interval)

            start = r_peak - window_size // 2
            end = r_peak + window_size // 2
            if start < 0 or end >= len(ecg_signal[lead]):
                continue
            heartbeats.append(ecg_signal[lead][start:end])

        average_heartbeat = np.mean(heartbeats, axis=0)
        average_baseline = np.median(average_heartbeat)
        average_rr_interval = np.mean(rr_intervals)

        if average_rr_interval//500 < 0.5:
            X = 0.04
        elif average_rr_interval//500 >= 0.5 and average_rr_interval//500 < 0.544:
            X = 0.048
        elif average_rr_interval//500 >= 0.548 and average_rr_interval//500 < 0.6:
            X = 0.056
        elif average_rr_interval//500 >=0.6:
            X = 0.064


        J_point=int(200+500*X)
        Reference_point=int(J_point+500*0.07)


        plt.subplot(311)
        plt.title(lead+1)
        plt.plot(ecg_signal[lead])
        plt.axhline(y=average_baseline, ls='--', c='blue')
        plt.scatter(real_r_peaks, ecg_signal[lead][real_r_peaks], color='r', label='R_peaks')


        plt.subplot(312)
        for heartbeat in heartbeats:
            plt.plot(heartbeat)

        plt.subplot(313)
        plt.plot(average_heartbeat)
        plt.axhline(y=average_baseline, ls='--', c='blue')
        plt.scatter(200, average_heartbeat[200], color='r',label='R_peaks')
        plt.scatter(Reference_point, average_heartbeat[Reference_point])
        plt.show()


        Reference_mV=average_heartbeat[Reference_point]


        if Reference_mV-average_baseline >0.1 :
            lead_label=1
            print(Reference_mV-average_baseline,'STE')

        elif average_baseline-Reference_mV>0.05:
            lead_label =1
            print(average_baseline-Reference_mV,'STD')
        else:
            lead_label = 0
            print('Normal')

        lead_labels.append(lead_label)

    return lead_labels




if __name__ == '__main__':
    ecg_signal = np.load('test_data.npy')
    lead_labels = beat_template(ecg_signal)
    print(lead_labels)



