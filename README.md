# ECG-Analyzer

Программы, облегчающие анализ ЭКГ при выполнении лабораторной работы на снятие электрокардиограммы (I и III отведений) и нахождение электрической оси сердца.

<!-- TOC -->
* [ECG-Analyzer](#ecg-analyzer)
  * [Основные возможности](#основные-возможности)
    * [ECGAnalyzer](#ecganalyzer)
    * [ElectricAxisCalculator](#electricaxiscalculator)
  * [Как использовать](#как-использовать)
    * [1. Установка](#1-установка)
    * [2. Использование](#2-использование)
      * [2.1. ECGAnalyzer](#21-ecganalyzer)
      * [2.2. ElectricAxisCalculator](#22-electricaxiscalculator)
<!-- TOC -->

## Основные возможности

### ECGAnalyzer
- Загрузка данных ЭКГ из CSV/TXT файлов
- Фильтрация сигналов:
  - Медианная фильтрация
  - Удаление базовой линии
  - Полосовая фильтрация (0.5-25 Гц)
  - Сглаживание фильтром Савицкого-Голея
- Детекция зубцов P, Q, R, S, T
- Расчет интервалов (PR, QRS, QT, ST, RR)
- Определение частоты сердечных сокращений (ЧСС)
- Расчет электрической оси сердца
- Визуализация сигналов в двух отведениях
- Генерация текстового отчета с параметрами ЭКГ

Данные при снятии (без обработки):\
![ECG_before](https://github.com/Tankolom-X/ECG-Analyzer/blob/main/media/ECG_before.jpg?raw=True "ECG_before")

После обработки программой:\
![ECG_after](https://github.com/Tankolom-X/ECG-Analyzer/blob/main/media/ECG_after.jpg?raw=True "ECG_after")

### ElectricAxisCalculator
- Расчет угла электрической оси сердца (α) по методу треугольника Эйнтховена
- Визуализация треугольника отведений и направления электрической оси
- Интерактивный ввод параметров ЭКГ (амплитуды зубцов Q, R, S I и III отведений)
- Автоматический расчет алгебраических сумм для I и III отведений

![ECG_axis](https://github.com/Tankolom-X/ECG-Analyzer/blob/main/media/ECG_axis.jpg?raw=True "ECG_axis")

## Как использовать
### 1. Установка

Вы можете скачать и запустить скомпилированные программы для вашей операционной системы.
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

> *Или запустить, загрузив код из источника. Для этого введите следующие команды в командной строке:*
   >   + Клонируйте репозиторий себе на устройство
   >   ```bash
   >   git clone https://github.com/Tankolom-X/ECG-Analyzer.git
   >   ```
   >   + Перейдите в папку с проектом
   >   ```bash
   >   cd ECG-Analyzer
   >   ```
   >   + Убедитесь, что у вас установлена актуальная версия pip:
   >   ```bash
   >   pip install --upgrade pip 
   >   ```
   >   + Затем установите требуемые зависимости:
   >   ```bash
   >   pip install -r requirements.txt 
   >   ```
   >   + Запустите нужный файл (ECGAnalyzer.py или ElectricAxisCalculator.py)
   >   ```bash
   >   python ECGAnalyzer.py
   >   ``` 

### 2. Использование
#### 2.1. ECGAnalyzer
При запуске программы Вас попросят указать полный путь до .txt или .csv файла, в котором находятся данные ЭКГ.\
Данные должны содержать 3 столбца, разделитель - ";":
- время (в мкс)
- I отведение (канал 1) (в мВ)
- III отведение (канал 2) (в мВ)

   > Экспорт данных при помощи программы LGraph2
   >  1. Открыть программу LGraph2
   >  2. Нажать "Файл" -> "Загрузить"
   >  3. Выбрать на устройстве файл с записанной ЭКГ в формате .par -> "Продолжить"
   >  4. Нажать "Файл" -> "Экспорт данных". Установить следующие настройки экспорта:\
   > Формат вывода - "Вольты"\
   > Разделитель между колонками - "Точка с запятой"\
   > Убрать галочку с пункта "Заголовок с параметрами ввода"\
   > Добавить столбец времени, размерность - "мкс"\
   > Тип файла - "Текстовый"\
   > Выбрать "Все каналы"
   >  5. Нажать "Экспортировать", выбрать имя файла и место сохранения

После этого появится окно с обработанной кардиограммой.\
При закрытии окна с кардиограммой в терминале отобразится текстовый отчет с основными параметрами ЭКГ.

#### 2.2. ElectricAxisCalculator
При запуске программы Вас попросят ввести значения амплитуд зубцов Q, R, S для I и III отведений ЭКГ.\
После ввода данных программа построит треугольник Эйнтховена и отобразит электрическую ось сердца.
