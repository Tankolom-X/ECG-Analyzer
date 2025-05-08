# ECG-Analyzer

Программы, облегчающие анализ ЭКГ при выполнении лабораторной работы на снятие 1 и 3 отведений и нахождение электрической оси сердца.

<table>
      <thead>
         <th>
            <p align="center">
               <a href="https://github.com/Tankolom-X/ECG-Analyzer/releases/latest/download/ECG-Analyzer_windows.zip" target="_blank">
                  <picture>
                     <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/windows_white.png">
                     <source media="(prefers-color-scheme: light)" srcset="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/windows.png">
                     <img alt="windows" src="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/windows.png">
                  </picture>
               </a>
            </p>
         </th>
         <th>
            <p align="center">
               <a href="https://github.com/Tankolom-X/ECG-Analyzer/releases/latest/download/ECG-Analyzer_macos.zip" target="_blank">
                  <picture>
                     <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/apple_white.png">
                     <source media="(prefers-color-scheme: light)" srcset="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/apple.png">
                     <img alt="apple" src="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/apple.png">
                  </picture>
               </a>
            </p>
         </th>
         <th>
            <p align="center">
               <a href="https://github.com/Tankolom-X/ECG-Analyzer/releases/latest/download/ECG-Analyzer_linux.zip" target="_blank">
                  <picture>
                     <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/linux_white.png">
                     <source media="(prefers-color-scheme: light)" srcset="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/linux.png">
                     <img alt="linux" src="https://github.com/Tankolom-X/CommitDraw/blob/1.x/media/os_icons/linux.png">
                  </picture>
               </a>
            </p>
         </th>
      </thead>
      <tbody>
         <th>
            <a href="https://github.com/Tankolom-X/ECG-Analyzer/releases/latest/download/ECG-Analyzer_windows.zip">Download</a>
         </th>
         <th>
            <a href="https://github.com/Tankolom-X/ECG-Analyzer/releases/latest/download/ECG-Analyzer_macos.zip">Download</a>
         </th>
         <th>
            <a href="https://github.com/Tankolom-X/ECG-Analyzer/releases/latest/download/ECG-Analyzer_linux.zip">Download</a>
         </th>
      </tbody>
   </table>

   <br>

## ECGAnalyzer
Позволяет визуализировать данные, полученные при снятии ЭКГ, и убрать часть помех, а также найти изолинию. Находит предварительные амплитуды зубцов и длительности сегментов.
Принимает на входе полный путь до .txt файла - результата экспорта данных ЭКГ.
Файл должен содержать информацию в csv формате с 3 столбцами - время (в мкс), 1 отведение (канал 1) (в мВ), 3 отведение (канал 2) (в мВ). Разделитель - ";". 

Данные при снятии (без обработки):\
![ECG_before](https://github.com/Tankolom-X/ECG-Analyzer/blob/main/media/ECG_before.jpg?raw=True "ECG_before")

После обработки программой:\
![ECG_after](https://github.com/Tankolom-X/ECG-Analyzer/blob/main/media/ECG_after.jpg?raw=True "ECG_after")

## ElectricAxisCalculator
Позволяет найти электрическую ось сердца по средним значениям амплитуд зубцов Q, R, S 1 и 3 отведений.\
![ECG_axis](https://github.com/Tankolom-X/ECG-Analyzer/blob/main/media/ECG_axis.jpg?raw=True "ECG_axis")