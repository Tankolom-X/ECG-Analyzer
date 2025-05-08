from idlelib.replace import replace
from os import remove

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, medfilt, sosfiltfilt, find_peaks, savgol_filter
import pandas as pd
from math import atan2, degrees
from typing import Dict, List
from matplotlib.gridspec import GridSpec


class ECGProcessor:
    def __init__(self, file_path: str, max_display_time: int = 60):
        self.file_path = file_path
        self.max_display_time = max_display_time
        self.fs = 250  # Частота дискретизации по умолчанию
        self.time_s = None
        self.lead_I = None
        self.lead_III = None
        self.ecg_analysis = None

    def load_data(self) -> None:
        """Загрузка и предварительная обработка данных ЭКГ (уже в мВ)"""
        try:
            df = pd.read_csv(self.file_path, sep=';', header=None,
                             names=['time_us', 'lead1', 'lead3'],
                             dtype=float, on_bad_lines='warn')
        except:
            with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            data = []
            for line in lines:
                try:
                    parts = line.replace(',', '.').strip().split(';')
                    if len(parts) == 3:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except:
                    continue
            df = pd.DataFrame(data, columns=['time_us', 'lead1', 'lead3'])

        if len(df) < 10:
            raise ValueError("Недостаточно данных для анализа")

        # Обработка и сортировка данных
        df = df.drop_duplicates('time_us').sort_values('time_us')
        self.time_s = df['time_us'].values / 1_000_000
        self.lead_I = df['lead1'].values
        self.lead_III = df['lead3'].values

        # Расчет частоты дискретизации
        if len(df) > 1:
            self.fs = 1_000_000 / np.median(np.diff(df['time_us'].values))

    def process_signals(self) -> None:
        """Обработка и фильтрация сигналов ЭКГ (данные уже в мВ)"""

        def normalize_and_filter(signal: np.ndarray) -> np.ndarray:
            """Фильтрация сигнала (только центрирование, без масштабирования)"""
            # Центрирование сигнала
            signal = signal - np.median(signal)

            # Каскадная фильтрация
            signal = medfilt(signal, 5)

            # Удаление базовой линии
            window_size = min(int(self.fs * 0.6), len(signal) - 1)
            if window_size % 2 == 0:
                window_size -= 1
            if window_size < 3:
                window_size = 3
            baseline = medfilt(signal, kernel_size=window_size)
            signal = signal - baseline

            # Полосовая фильтрация 0.5-25 Гц
            nyq = 0.5 * self.fs
            sos = butter(4, [0.5 / nyq, 25 / nyq], btype='bandpass', output='sos')
            signal = sosfiltfilt(sos, signal)

            # Дополнительное сглаживание
            if len(signal) > 30:
                window = min(31, len(signal))
                if window % 2 == 0:
                    window -= 1
                signal = savgol_filter(signal, window, 3)

            return signal

        self.lead_I = normalize_and_filter(self.lead_I)
        self.lead_III = normalize_and_filter(self.lead_III)

    def detect_waveforms(self, signal: np.ndarray, peaks: np.ndarray) -> Dict[str, List[float]]:
        """Улучшенная детекция всех зубцов и интервалов для данных в мВ"""
        waveforms = {
            'P': [], 'Q': [], 'R': [], 'S': [], 'T': [],
            'P_dur': [], 'Q_dur': [], 'R_dur': [], 'S_dur': [], 'T_dur': [],
            'PR': [], 'QRS': [], 'QT': [], 'ST': [], 'RR': []
        }

        for i, peak in enumerate(peaks):
            if i == 0:
                continue  # Пропускаем первый цикл

            prev_peak = peaks[i - 1]
            current_peak = peak

            # 1. Расчет RR интервала
            rr_interval = (current_peak - prev_peak) / self.fs
            waveforms['RR'].append(rr_interval)

            # 2. Детекция QRS комплекса
            q_search_start = max(prev_peak + int(0.1 * self.fs), current_peak - int(0.15 * self.fs))
            q_search_end = current_peak - int(0.02 * self.fs)
            q_window = signal[q_search_start:q_search_end]

            q_candidates = np.where(q_window < np.percentile(q_window, 10))[0]
            q_point = q_search_start + (q_candidates[0] if len(q_candidates) > 0 else np.argmin(q_window))

            # Поиск S точки
            s_search_start = current_peak + int(0.02 * self.fs)
            s_search_end = min(len(signal) - 1, current_peak + int(0.15 * self.fs))
            s_window = signal[s_search_start:s_search_end]

            s_candidates = np.where(s_window < np.percentile(s_window, 10))[0]
            s_point = s_search_start + (s_candidates[0] if len(s_candidates) > 0 else np.argmin(s_window))

            # Длительности компонентов QRS
            q_duration = (current_peak - q_point) / self.fs
            r_duration = 0.04
            s_duration = (s_point - current_peak) / self.fs
            qrs_duration = (s_point - q_point) / self.fs

            # 3. Детекция P волны
            p_search_start = max(0, q_point - int(0.3 * self.fs))
            p_search_end = q_point - int(0.02 * self.fs)

            p_point = p_search_start + np.argmax(np.abs(signal[p_search_start:p_search_end]))

            p_level = np.median(signal[p_search_start:p_search_end])
            p_crossings = np.where(np.diff(np.sign(signal[p_search_start:p_search_end] - p_level)))[0]

            if len(p_crossings) >= 2:
                p_start = p_search_start + p_crossings[0]
                p_end = p_search_start + p_crossings[-1]
            else:
                p_start = p_point - int(0.04 * self.fs)
                p_end = p_point + int(0.04 * self.fs)

            p_duration = (p_end - p_start) / self.fs

            # 4. Детекция T волны
            t_search_start = s_point + int(0.05 * self.fs)
            t_search_end = min(len(signal) - 1, s_point + int(0.4 * self.fs))

            t_point = t_search_start + np.argmax(np.abs(signal[t_search_start:t_search_end]))

            t_level = np.median(signal[t_search_start:t_search_end])
            t_crossings = np.where(np.diff(np.sign(signal[t_search_start:t_search_end] - t_level)))[0]

            if len(t_crossings) >= 2:
                t_start = t_search_start + t_crossings[0]
                t_end = t_search_start + t_crossings[-1]
            else:
                t_start = t_point - int(0.06 * self.fs)
                t_end = t_point + int(0.06 * self.fs)

            t_duration = (t_end - t_start) / self.fs

            # 5. Расчет интервалов
            pr_interval = (q_point - p_start) / self.fs
            qt_interval = (t_end - q_point) / self.fs
            st_segment = (t_start - s_point) / self.fs

            # 6. Сохранение параметров
            waveforms['P'].append(signal[p_point])
            waveforms['Q'].append(signal[q_point])
            waveforms['R'].append(signal[current_peak])
            waveforms['S'].append(signal[s_point])
            waveforms['T'].append(signal[t_point])

            waveforms['P_dur'].append(p_duration)
            waveforms['Q_dur'].append(q_duration)
            waveforms['R_dur'].append(r_duration)
            waveforms['S_dur'].append(s_duration)
            waveforms['T_dur'].append(t_duration)

            waveforms['PR'].append(pr_interval)
            waveforms['QRS'].append(qrs_duration)
            waveforms['QT'].append(qt_interval)
            waveforms['ST'].append(st_segment)

        return waveforms

    def analyze_ecg(self) -> None:
        """Анализ параметров ЭКГ"""
        min_length = min(len(self.time_s), len(self.lead_I), len(self.lead_III))
        mask = (self.time_s[:min_length] >= 0) & (self.time_s[:min_length] <= self.max_display_time)
        time_window = self.time_s[:min_length][mask]
        lead_I_window = self.lead_I[:min_length][mask]
        lead_III_window = self.lead_III[:min_length][mask]

        def find_isoline(signal):
            hist, bins = np.histogram(signal, bins=50)
            return bins[np.argmax(hist)]

        isoline_I = find_isoline(lead_I_window)
        isoline_III = find_isoline(lead_III_window)

        lead_I_window = lead_I_window - isoline_I
        lead_III_window = lead_III_window - isoline_III

        peaks, _ = find_peaks(lead_I_window,
                              height=np.percentile(lead_I_window, 90),
                              distance=int(self.fs * 0.3),
                              prominence=0.15)  # Порог уменьшен для мВ

        if len(peaks) < 2:
            self.ecg_analysis = None
            return

        waveforms_I = self.detect_waveforms(lead_I_window, peaks)
        waveforms_III = self.detect_waveforms(lead_III_window, peaks)

        def calculate_averages(waveforms):
            return {k: np.mean(v) if v else 0 for k, v in waveforms.items()}

        avg_waveforms_I = calculate_averages(waveforms_I)
        avg_waveforms_III = calculate_averages(waveforms_III)

        hr = 60 / np.mean(waveforms_I['RR']) if waveforms_I['RR'] else 0

        avg_qrs_I = np.mean(waveforms_I['R']) if waveforms_I['R'] else 0
        avg_qrs_III = np.mean(waveforms_III['R']) if waveforms_III['R'] else 0
        axis = degrees(atan2(avg_qrs_III, avg_qrs_I))

        rr_std = np.std(waveforms_I['RR']) if waveforms_I['RR'] else 0
        if rr_std > 0.15 * np.mean(waveforms_I['RR']) if waveforms_I['RR'] else 1:
            if hr > 100:
                rhythm_status = "Тахиаритмия"
            elif hr < 60:
                rhythm_status = "Брадиаритмия"
            else:
                rhythm_status = "Нерегулярный ритм"
        else:
            rhythm_status = "Нормальный ритм"

        self.ecg_analysis = {
            'time_window': time_window,
            'lead_I': lead_I_window,
            'lead_III': lead_III_window,
            'peaks': peaks,
            'hr': hr,
            'heart_axis': axis,
            'rhythm_status': rhythm_status,
            'isoline_I': isoline_I,
            'isoline_III': isoline_III,
            'rr_std': rr_std,
            'waveforms_I': waveforms_I,
            'waveforms_III': waveforms_III,
            'avg_waveforms_I': avg_waveforms_I,
            'avg_waveforms_III': avg_waveforms_III
        }

    def plot_results(self) -> None:
        """Визуализация результатов анализа"""
        if not self.ecg_analysis:
            print("Недостаточно данных для построения графиков")
            return

        plt.style.use('default')

        fig = plt.figure(figsize=(16, 10), facecolor='#f5f5f5')
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.4)

        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = '#e0e0e0'
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['axes.edgecolor'] = '#d0d0d0'

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.ecg_analysis['time_window'],
                 self.ecg_analysis['lead_I'],
                 label='Отведение I', color='#1f77b4', linewidth=1.2)
        ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

        for peak in self.ecg_analysis['peaks']:
            if peak < len(self.ecg_analysis['time_window']):
                ax1.plot(self.ecg_analysis['time_window'][peak],
                         self.ecg_analysis['lead_I'][peak],
                         'ro', markersize=4, alpha=0.7)

        ax1.set_title(f'ЭКГ - Отведение I (ЧСС: {self.ecg_analysis["hr"]:.1f} уд/мин)', pad=15)
        ax1.set_xlabel('Время (с)', fontsize=10)
        ax1.set_ylabel('Амплитуда (мВ)', fontsize=10)
        ax1.tick_params(axis='both', which='major', labelsize=9)
        ax1.legend(loc='upper right', fontsize=9)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.ecg_analysis['time_window'],
                 self.ecg_analysis['lead_III'],
                 label='Отведение III', color='#2ca02c', linewidth=1.2)
        ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)


        ax2.set_xlabel('Время (с)', fontsize=10)
        ax2.set_ylabel('Амплитуда (мВ)', fontsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=9)
        ax2.legend(loc='upper right', fontsize=9)

        fig.suptitle('Анализ ЭКГ сигнала', y=0.98, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def print_report(self) -> None:
        """Вывод подробного текстового отчета по анализу"""
        if not self.ecg_analysis:
            print("Не удалось провести анализ - недостаточно данных")
            return

        def format_param(value, unit):
            return f"{value:.2f} {unit}" if isinstance(value, (int, float)) else str(value)

        print("\n" + "=" * 50)
        print(" " * 15 + "РЕЗУЛЬТАТЫ АНАЛИЗА ЭКГ")
        print("=" * 50)

        print("\n[ОСНОВНЫЕ ПАРАМЕТРЫ]")
        print(f"{'Частота сердечных сокращений:':<35} {format_param(self.ecg_analysis['hr'], 'уд/мин')}")
        print(f"{'Электрическая ось сердца:':<35} {format_param(self.ecg_analysis['heart_axis'], '°')}")
        print(f"{'Характер ритма:':<35} {self.ecg_analysis['rhythm_status']}")
        print(f"{'Вариабельность RR-интервалов:':<35} {format_param(self.ecg_analysis['rr_std'] * 1000, 'мс')}")

        print("\n[ОТВЕДЕНИЕ I]")
        print("Амплитуды:")
        print(f"  {'P:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['P'], 'мВ')}")
        print(f"  {'Q:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['Q'], 'мВ')}")
        print(f"  {'R:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['R'], 'мВ')}")
        print(f"  {'S:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['S'], 'мВ')}")
        print(f"  {'T:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['T'], 'мВ')}")

        print("\nДлительности:")
        print(f"  {'P:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['P_dur'] * 1000, 'мс')}")
        print(f"  {'Q:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['Q_dur'] * 1000, 'мс')}")
        print(f"  {'R:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['R_dur'] * 1000, 'мс')}")
        print(f"  {'S:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['S_dur'] * 1000, 'мс')}")
        print(f"  {'T:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['T_dur'] * 1000, 'мс')}")

        print("\nИнтервалы:")
        print(f"  {'PR:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['PR'] * 1000, 'мс')}")
        print(f"  {'QRS:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['QRS'] * 1000, 'мс')}")
        print(f"  {'QT:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['QT'] * 1000, 'мс')}")
        print(f"  {'ST:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['ST'] * 1000, 'мс')}")
        print(f"  {'RR:':<5} {format_param(self.ecg_analysis['avg_waveforms_I']['RR'] * 1000, 'мс')}")

        print("\n[ОТВЕДЕНИЕ III]")
        print("Амплитуды:")
        print(f"  {'P:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['P'], 'мВ')}")
        print(f"  {'Q:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['Q'], 'мВ')}")
        print(f"  {'R:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['R'], 'мВ')}")
        print(f"  {'S:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['S'], 'мВ')}")
        print(f"  {'T:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['T'], 'мВ')}")

        print("\nДлительности:")
        print(f"  {'P:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['P_dur'] * 1000, 'мс')}")
        print(f"  {'Q:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['Q_dur'] * 1000, 'мс')}")
        print(f"  {'R:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['R_dur'] * 1000, 'мс')}")
        print(f"  {'S:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['S_dur'] * 1000, 'мс')}")
        print(f"  {'T:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['T_dur'] * 1000, 'мс')}")

        print("\nИнтервалы:")
        print(f"  {'PR:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['PR'] * 1000, 'мс')}")
        print(f"  {'QRS:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['QRS'] * 1000, 'мс')}")
        print(f"  {'QT:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['QT'] * 1000, 'мс')}")
        print(f"  {'ST:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['ST'] * 1000, 'мс')}")
        print(f"  {'RR:':<5} {format_param(self.ecg_analysis['avg_waveforms_III']['RR'] * 1000, 'мс')}")

        print("\n" + "=" * 50 + "\n")

    def run_analysis(self) -> None:
        """Полный цикл анализа ЭКГ"""
        try:
            print(f"\nЗагрузка файла: {self.file_path}")
            self.load_data()
            print(f"  • Обнаружено {len(self.time_s)} точек ({self.time_s[-1]:.1f} сек)")
            print(f"  • Частота дискретизации: {self.fs:.1f} Гц")

            print("\nОбработка сигналов...")
            self.process_signals()
            self.analyze_ecg()

            print("\nВизуализация результатов...")
            self.plot_results()

            print("\nФормирование отчета...")
            self.print_report()

        except Exception as e:
            print(f"\nОшибка при анализе: {str(e)}")
            raise


if __name__ == "__main__":
    path = input("Укажите полный путь до файла .txt: ")
    path = path.replace("\"", "")
    ecg_processor = ECGProcessor(path)
    ecg_processor.run_analysis()