
# PQ ONE Beginner Operation Guide

This guide is designed for first-time users of PQ ONE. It walks you through core operations like loading measurement data, viewing trend graphs, analyzing events, and exporting reports — with clear steps and practical tips.

---

## 1. Launching and Loading Data

### 1.1 Launching PQ ONE

1. Locate the `PQ ONE` shortcut on your desktop or start menu.
2. Double-click to launch the application.
3. Wait for the main interface to load — it includes a toolbar, trend graph area, and event list pane.

> 💡 Tip: If PQ ONE does not open, check that .NET Framework 4.5.2 or higher is installed.

---

### 1.2 Loading Data from Local Folder

1. Click the **Folder** icon on the top toolbar.
2. In the dropdown menu, select **Open Folder**.
3. In the dialog box, navigate to the folder containing your `.pqd` or `.pqdif` measurement files.
4. Select the file and click **Open**.
5. PQ ONE will load the data and display the corresponding trend graphs and event list.


![Example: Open Data Folder](assets/open_folder_menu2.png)
![Example: Open Data Folder](assets/open_folder_menu1.png)

> ⚠ **Caution**: Do not delete, rename, or move the data folder while it's open in PQ ONE. Doing so may cause the application to crash.

>  **Tip**: You can also drag & drop a folder directly into the PQ ONE window to load it.
---


### 1.3 Loading Recent Measurement Data

PQ ONE remembers up to 10 of the most recently loaded measurement files. This feature allows you to quickly reopen data you worked on recently.

#### Steps:

1. Launch PQ ONE. On the home screen, locate the **Recent the measurement data** section.
2. You will see folders representing recently accessed measurement datasets.
3. Simply **double-click** on a folder to reopen it.

Alternatively, you can:

- Click the **Folder** icon on the toolbar.
- From the dropdown, select **Recent the measurement data**, then choose a file from the list.

![Recent Measurement Data Screen](assets/recent_measurements.png)

>  **Tip**: This is useful when switching between multiple datasets during analysis.

> ⚠ **Note**: If a folder shown in the recent list has been moved or deleted, PQ ONE may display an error when attempting to open it.


## 2. Using the Data List

The Data List feature in PQ ONE allows users to register, view, and manage measurement data folders. You can drag-and-drop folders, open multiple datasets, and use filters for repeated data. This section explains all available methods.

---

### 2.1 Registering Data by Drag and Drop

You can drag and drop a folder containing measurement data directly into the PQ ONE window.

- PQ ONE will automatically detect and attempt to register the data.
- A prompt will appear asking whether to **Add to the data list** or **Refresh**.

![Drag and Drop Confirmation DialogConfirmation Dialog](assets/drag_drop_prompt.png)

**Options in the dialog:**

- **Yes** – Add the folder to the current list.
- **No** – Clear the current list and register only this folder.
- **Cancel** – Do not load anything.
![Dialog](assets/yes.png)
> 💡 Tip: You can drop a folder that contains multiple subfolders — PQ ONE will try to detect all valid datasets inside.

---

### 2.2 Displaying the Data List Screen

To manually open the data list screen:

1. From the top menu, click `View ▶ Data List`
2. Alternatively, click the **Data list** icon on the toolbar.
3. Select one or more folders to load, then press **OK**.

![Data List Overview](assets/data_list_screen.png)

> ⚠ Note: Only valid data folders will appear in the list.

---

### 2.3 Reading Repeated Measurement Data

If you have repeated measurements (e.g., from the same test setup across different dates), you can read them together:

1. In the Data List screen, right-click on a folder.
2. Choose **Select the check boxes for repeated data**.
3. All matching repeated datasets will be selected automatically.

![Repeated Data Context Menu](assets/repeated_data_checkboxes.png)

####  Repeated Data Matching Conditions

- Same **Model**, **Wiring**, **Frequency**, **Uref**, **Trend Interval**, and **Recording Items**
- Folder name follows repeated format (e.g., `20230101_01`, `20230101_02`)
- Measurement periods **do not overlap**

>  Matched folders are highlighted with the same background color.

---

### 2.4 Reading Data via [Open Folder] Menu

You can also load data via the **Open Folder** menu:

1. From the toolbar, click the **Folder** icon ▶ **Open Folder**
2. In the folder browser, select the measurement data folder.
3. If it contains multiple valid datasets, the **Data List Screen** will appear automatically.

![Folder Browser Window](assets/folder_browser.png)

> Use this method when working with raw folders received from instruments or copied from USB drives.

### 2.5 Read data imported by GENNECT One
HIOKI "GENNECT One" application is a free application software that allows centralized management
of multiple measurement devices in a LAN, automatic transfer of measurement files, and remote
control.
* Please refer to GENNECT One instruction manual for how to import data to GENNECT One.

## 3. Understanding the Main Screen Layout

Once you load measurement data into PQ ONE, the main interface displays several key components that work together to help you analyze and interpret the data. This section describes each part of the screen and how to adjust the layout.

---

### 3.1 Overview of Main Interface Areas
![Main Screen Layout Overview](assets/main_screen_layout.png)  
The PQ ONE main screen includes the following five key areas:
#### Area ① – Toolbar
The toolbar at the top provides quick access to common actions. From left to right, it includes:
- Opening a data folder
- Setting measurement options
- Loading saved configuration files
- Exporting reports in Word format
- Converting data to CSV
- Accessing the statistics and standards view
- Opening the instruction manual

You can hover over each icon to see a tooltip explaining its function.

#### Area ② – Period Selector

This section lets you define the period (start and end time) for trend graph visualization. It helps you zoom in on a specific time window of the measurement data.

#### Area ③ – Trend Graph

The center of the screen displays the **Trend Graph**, which shows line plots of voltage, current, frequency, and other key measurements over time. This is the main tool for analyzing long-term patterns or identifying unusual fluctuations.

#### Area ④ – Event List

On the left side of the screen, the **Event List** summarizes all recorded events such as voltage dips, swells, transients, or interruptions. It includes timestamps, event types, affected channels, and severity levels. Events can be selected to view their corresponding waveform.

#### Area ⑤ – Event Data Tabs

At the bottom of the screen, you will find tabs for viewing detailed data of the selected event. These include:
- **Event Waveform**: Raw waveform around the event
- **Harmonics**: Harmonic spectrum
- **Vector**: Phasor diagram
- **DMM**: RMS snapshot values

Each tab gives a different perspective on the event characteristics.

---

### 3.2 Resizing and Resetting the Layout

PQ ONE allows you to freely adjust the screen layout:

- You can drag the **split bars** between sections to resize the graph, event list, or waveform areas vertically and horizontally.
- The layout will be remembered between sessions, so your preferred arrangement is retained.

If you want to reset the layout to its original form:

1. Right-click on the toolbar.
2. Choose **Initialize the display** from the context menu.

This will restore all panel sizes and positions to their default layout.

![Change Display Size](assets/change_display_size.png)

### 3.3 Viewing the trend graph

#### Selecting the period

Select the period you wish to display in the window under **[Period]** on the top right of the window.  
The period setting applies to all trend graph tabs.

![period-select-1](assets/period-select-1.png)
![period-select-2](assets/period-select-2.png)
![period-select-3](assets/period-select-3.png)
![period-select-4](assets/period-select-4.png)
- **[Select Period]** allows you to set the display start time (**From**) and the display end time (**To**) for trend graphs.
- Other values (e.g., "1 Week") allow you to set only the start time (**From**) and automatically set the end time.

####  Zooming in on the trend graph

You can zoom in on the selected portion of the trend graph by dragging the mouse:

- **Drag diagonally** to zoom in on both the **Y-axis** and **X-axis**.
- **Drag vertically** to zoom in on only the **Y-axis**.
- **Drag horizontally** to zoom in on only the **X-axis**.

![zoom-diagonal](assets/zoom-diagonal.png)
![zoom-vertical-horizontal](assets/zoom-vertical-horizontal.png)
![zoom-horizontal](assets/zoom-horizontal.png)

####  Adjust the range of the X-axis on the trend graph

- Click the magnifying glass icon ![Zoom In icon] to zoom **in** on the X-axis range.
- Click the magnifying glass icon ![Zoom Out icon] to zoom **out** on the X-axis range.
- Move the mouse cursor over the graph to display the zoom area, centered around the cursor position.

> *Maximum zoom-in range of the X-axis is 12 points.*

![x-axis-adjust](assets/x-axis-adjust.png)

#### Adjusting the range of the Y-axis on the trend graph

Clicking the Y-axis auto-scale icon ![Y auto-scale icon] automatically adjusts the Y-axis range so that the maximum and minimum values for the selected period fit in the graph.

Clicking the icon successively cycles through the following scale modes:

1. **First click**: Adds a margin of approximately ±10% to the fluctuation range.  
   (If the lower limit is 0, only positive margin is applied.)

 

2. **Second click**: Adds a margin of approximately ±100% to the fluctuation range.  
   (Again, if the lower limit is 0, only positive margin is applied.)

  ![y-scale-click](assets/y-scale-first-click.png)

3. **Third click**: Fixes the graph from 0 to the maximum value, ignoring fluctuation range.

   ![y-scale-third-click](assets/y-scale-third-click.png)
  
Clicking the button again returns to the initial mode (1).

---

####  Y-axis display range specification

To manually specify the Y-axis display range:

- **Double-click** the Y-axis, or  
- Right-click and choose **[Vertical Axis Setting]** from the context menu.

A dialog will open where you can set the **Min** and **Max** values manually.

> The min/max values are automatically suggested based on the current voltage/current range.

![y-axis-setting-dialog](assets/y-axis-setting-dialog.png)

---

#### Displaying the entire trend graph

Click the **Full Graph View** icon ![Full graph view icon] to revert the graph view to its initial state.  
This cancels any zoom settings and restores the full display of both **X** and **Y** axes.

![full-graph-before and after-click](assets/full-graph.png)  

####  Displaying measured values

You can click inside the trend graph to display a **vertical cursor line**. This cursor shows the exact measured values (voltage and current) at the selected timestamp.

- The cursor appears with numerical values at the right side of the screen.
- Click anywhere else inside the graph to move the cursor.
- Clicking outside the graph area hides the cursor.

For example:
- At time `8/14/2016 15:56:10.040`:
  - U1 rms AVG: 204.42 V
  - U2 rms AVG: 204.52 V
  - U3 rms AVG: 205.53 V
  - I1 rms AVG: 36.89 A
  - I2 rms AVG: 66.39 A
  - I3 rms AVG: 56.77 A

![display-cursor](assets/display-cursor.png)
---

#### Selecting display parameters

Measurement parameters are organized by tabs (e.g., “U/I”, “Frequency”, “Power”, etc.).  
You can select parameters from each tab via a **drop-down menu**.

- Up to 3 trend graphs can be displayed at the same time.
- Selecting “[-]” will hide that graph tab.

![dropdown-menu](assets/dropdown-menu.png)

---

#### Supported parameter tabs and examples:

1. **Detail Trend Tab**
   - `U rms1/2`: RMS voltage per half-cycle
   - `I rms1/2`: RMS current per half-cycle
   - `Inrush`: Inrush current
   - `Freq_wav`: Frequency (1 waveform)
   - `Pinst`: Instantaneous flicker

2. **U / I Tab**
   - `U rms`: RMS voltage (200ms)
   - `U pk+, U pk-`: Voltage waveform peak
   - `U dc`: DC voltage
   - `U cf`: Voltage crest factor
   - `I rms`: RMS current (200ms)
   - `I pk+, I pk-`: Current waveform peak
   - `I dc`: DC current
   - `I cf`: Current crest factor
   - `U avg`: RMS voltage (average)
   - `I avg`: RMS current (average)

3. **Frequency Tab**
   - `Freq`: Frequency (200ms)
   - `f10s / Freq10s`: Frequency (10s)

4. **Unbalance Tab**
   - `Uunb0 / Iunb0`: Zero-phase unbalance
   - `U zero / neg / pos`: Voltage components
   - `I zero / neg / pos`: Current components

5. **Harmonics Tab**
   - `Uthd / Ithd`: Total harmonic distortion
   - `U harm / I harm / P harm`: Harmonics
   - `UharmH / IharmH`: High-order harmonics
   - `KF`: K factor
   - `TDD`: Total demand distortion *(→ see Chapter 4, p.65)*

6. **Flicker Tab**
   - `Pst / Plt`: Short/long-term flicker
   - `ΔV10 1min, 1hMAX, 1h4th, 1hAVG`: Voltage variation metrics

7. **Power Tab**
   - `P / S / Q`: Active, apparent, reactive power
   - `PF / DPF`: Power factor / Displacement PF
   - `Eff`: Power efficiency

8. **Energy Tab**
   - `WP+, WP-`: Active energy
   - `WQlag, WQlead`: Reactive energy
   - `WS`: Apparent energy
   - `Ecost`: Energy cost

9. **Demand Tab**
   - `Dem P+, P- / Qlag, Qlead / S`: Power demand values
   - `Dem WP+, WP- / WQlag, WQlead`: Energy demand
   - `Dem PF`: Demand power factor

10. **Mains Signaling Tab**
    - `Msv1 / Msv2`: Mains signaling voltage levels *(→ see p.36)*

---

#### Channel and checkbox selection

After choosing a parameter, additional **checkboxes** will appear:

- Select the channels (CH1 / CH2 / CH3) and attributes (MAX, AVG, MIN) to display on the graph.
- If no checkboxes appear, the graph is rendered immediately after parameter selection.

![checkbox-graph](assets/checkbox-graph.png)

---

#### Highlighting a parameter

You can click on a parameter’s name in the **legend** to highlight it using a bold/thick line.  
Click again to remove the highlight.

![highlight-example](assets/highlight-example.png)


#### [Detailed Trend] Tab

The **Detailed Trend** tab is used to visualize the fluctuation range of measured values within each recording interval.  
This fluctuation is shown as a **vertical line** connecting the **maximum** and **minimum** values during each interval.

---

##### Function

- This tab helps users identify how much a voltage, current, or frequency parameter fluctuates over time.
- It is useful for detecting unstable or spiking behavior in power systems.

---

#####  Supported Parameters by Device Model

For **PQ3100**, the selectable parameters are:
- `U rms1/2`: RMS voltage per half-cycle
- `I rms1/2`: RMS current per half-cycle
- `Inrush`: Inrush current
- `Freq_wav`: Frequency per waveform
- `Pinst`: Instantaneous flicker

For **PW3198 / PQ3198**, the selectable parameters are:
- `U rms1/2`: RMS voltage per half-cycle
- `I rms1/2`: RMS current per half-cycle
- `Inrush`: Inrush current
- `Pinst`: Instantaneous flicker

---

##### Cursor Value Explanation

When a cursor is placed on the trend graph, two values are displayed for each parameter:

- **Top value** = Maximum during the interval
- **Bottom value** = Minimum during the interval

Example:
- `U1 rms1/2`: Max = 207.05 V, Min = 204.07 V  
- `I2 rms1/2`: Max = 10.14 A, Min = 9.26 A  
- `Freq_wav`: Max = 60.069 Hz, Min = 59.860 Hz

---

##### Screenshot Example

![detailed-trend-tab](assets/detailed-trend-tab.jpg)


#### [U / I] Tab

The **U / I tab** displays trend graphs for voltage and current parameters.

You can check the **maximum**, **minimum**, and **average** values for each channel by toggling the checkboxes at the top-left: `[MAX]`, `[AVG]`, `[MIN]`.

##### Selectable parameters:

For **PQ3100**:
- `U rms`: Root-mean-square voltage
- `U pk+`: Positive peak voltage
- `U pk-`: Negative peak voltage
- `U dc`: DC component of voltage
- `U cf`: Voltage crest factor
- `I rms`: Root-mean-square current
- `I pk+`: Positive peak current
- `I pk-`: Negative peak current
- `I dc`: DC component of current
- `I cf`: Current crest factor
- `U avg`: Average voltage
- `I avg`: Average current

For **PW3198 / PQ3198**:
- Same as above.
- `U dc` and `I dc` are **only measured on CH4** when **DC setting is enabled**.

##### Measurement example:

At a given time:
- `U1 rms AVG` = 205.92 V
- `I2 rms AVG` = 9.71 A
- `U ave AVG` = 205.44 V

![ui-tab-example](assets/ui-tab-example.jpg)
---

#### ■ [Frequency] Tab

The **Frequency tab** displays trend graphs related to system frequency variation.

##### Selectable parameters:

For **PQ3100**:
- `Freq`: Frequency sampled every 200 ms
- `Freq10s`: Frequency sampled every 10 seconds

For **PW3198 / PQ3198**:
- `Freq`: Frequency (200 ms)
- `f10s`: Frequency (10 s)  
  > Note: “f10s” label appears starting from firmware version **V9.00**

##### Frequency example:

- `Freq AVG`: 59.978 Hz  
- `Freq10s AVG`: 59.995 Hz

This tab is ideal for identifying abnormal frequency deviation or grid instability.

![frequency-tab-example](assets/frequency-tab-example.jpg)
---

####  [Unbalance] Tab

The **Unbalance tab** displays voltage and current unbalance data using trend graphs.

It helps you assess power quality by visualizing phase imbalance and zero/negative/positive components.

##### Selectable parameters (common for PQ3100, PW3198, PQ3198):

- `Uunb0`: Zero-phase voltage unbalance factor
- `Uunb`: Negative-phase voltage unbalanc

#### ■ [Harmonics] Tab

The Harmonics tab is divided into two sub-tabs: **Trend** and **Peak Level**.  
It displays harmonic-related parameters for voltage, current, and power.

---

### [Trend] Tab

This tab shows trend graphs of **THD (Total Harmonic Distortion)** and other harmonic components.  
You can select from multiple parameters depending on your model.

#### Supported parameters by model:

For **PQ3100**:
- `U thd-f / thd-r`: Voltage THD (based on fundamental or RMS)
- `I thd-f / thd-r`: Current THD
- `U harm`: Voltage harmonics
- `I harm`: Current harmonics
- `P harm`: Power harmonics
- `KF`: K-factor
- `U iharm`: Interharmonic voltage
- `I iharm`: Interharmonic current
- `TDD`: Total Demand Distortion

For **PW3198 / PQ3198**:
- Same parameter set as PQ3100
- Additional: `U harmH`, `I harmH` (High-order harmonics)

> For more about TDD, refer to Chapter 4 “Setting options” (→ p.65)

#### Display types:

- Harmonic values can be shown as:
  - **Level**: Magnitude
  - **%fnd**: Percentage of fundamental
  - **Phase**: Angle (available only as AVG)

#### Display note:

- For parameters like `U harm`, `I harm`, `P harm`, average value (AVG) is displayed.
- MAX and MIN are not available even if selected.
- For interharmonics (`U iharm`, `I iharm`), you can select between level and %fnd.

#### Custom display order:

Clicking the **[Order]** button will open a dialog to select the harmonic or interharmonic order to display.

![harmonics-trend](assets/harmonics-trend.jpg)

---

### [Peak Level] Tab

The Peak Level tab shows a bar chart of **maximum harmonic levels** for each order within the selected interval.

#### How it works:

- The interval defined in the [Period] setting is used to compute peak values.
- You can see:
  - **AVG Peak**: Average peak value during interval
  - **MAX Peak**: Highest peak during interval
- The bar chart displays results per phase (e.g., U1, U2, U3).

#### Interaction:

- Click on a bar in the chart to show a **cursor**.
- The corresponding numeric values will appear in the list on the right.
- You can zoom into a specific range by dragging your mouse horizontally.
- Click **[Order]** to choose which harmonic orders to display.

![peak-level-bar](assets/peak-level-bar.jpg)

---

### Zooming and Resetting

- You can zoom in by dragging the mouse over a selected area of the bar graph.

![zooming-bar](assets/zooming-bar.jpg)

- Click the **Y auto-scale** button to automatically fit the bar graph in the visible area.
- Click the **Zoom reset** button to return to default display (0th to 50th order).


#### [Flicker] Tab

The Flicker tab displays trend graphs for voltage fluctuation-related parameters.

##### Selectable parameters (PQ3100 / PW3198 / PQ3198):

- `Pst`: Short-term flicker severity
- `Plt`: Long-term flicker severity (2 hours)
- `Plt10min`: Long-term flicker severity (10 minutes)
- `ΔV10`: Voltage fluctuation magnitude in different time scales:
  - `1min`: 1-minute value
  - `1hMAX`: 1-hour maximum
  - `1h4th`: 1-hour 4th largest
  - `1hAVG`: 1-hour average

##### Display usage:

- `Pst` and `Plt` help evaluate discomfort due to flicker.
- `ΔV10` helps quantify voltage instability within time windows.

Example:  
- U1 1min = 0.168 V  
- U1 1h4th = 0.177 V  
- U1 Plt = 0.254

![flicker-tab-ΔV10](assets/flicker-tab-ΔV10.jpg)  
![flicker-tab-pstplt](assets/flicker-tab-pstplt.jpg)

---

#### [Power] Tab

The Power tab displays trend graphs of active, reactive, and apparent power.

##### Selectable parameters:

- **PQ3100 / PW3198**:
  - `P`: Active power
  - `S`: Apparent power
  - `Q`: Reactive power
  - `PF / DPF`: Power factor / Displacement power factor

- **PQ3198** only:
  - `Eff`: Power efficiency

##### Display usage:

- Power curves can be shown as MAX, AVG, MIN.
- PF and DPF help assess power factor behavior.

Example:  
- `P sum AVG` = 2.53 kW  
- `PF sum MIN` = 0.8632

![power-tab-example](assets/power-tab-example.jpg)

---

#### [Energy] Tab

The Energy tab displays cumulative electrical energy and cost over time.

##### Selectable parameters:

- **PQ3100**:
  - `WP+ / WP-`: Active energy (+/-)
  - `WQlag / WQlead`: Reactive energy (lag/lead)
  - `WS`: Apparent energy
  - `Ecost`: Energy cost

- **PQ3198 / PW3198**:
  - Same parameters, except `WS` and `Ecost` may vary

##### Display usage:

- Cumulative line graphs show increase over time.
- `Ecost` provides cost estimation for consumed energy.

Example:  
- `WP+` = 13.7795 MWh  
- `Ecost` = 275.590 kYEN

![energy-tab-example](assets/energy-tab-example.jpg)


#### [Demand] Tab

The Demand tab displays bar graphs for demand value and demand quantity.

##### Selectable parameters:

- **PQ3100**:
  - `Dem P+`, `Dem P-`: Active power demand (+/-)
  - `Dem Qlag / Qlead`: Reactive power demand
  - `Dem S`: Apparent power demand
  - `Dem PF`: Power factor demand
  - `Dem WP+ / WP-`: Energy demand (+/-)
  - `Dem WQlag / WQlead`: Reactive energy demand
  - `Dem WS`: Apparent energy demand

- **PQ3198 / PW3198**:
  - `Dem P+`, `Dem P-`, `Dem Qlag`, `Dem Qlead`

##### Time resolution:

You can choose demand period via dropdown menu:
- `15min`: Demand bars for 0–15, 15–30, 30–45, 45–60 min
- `30min`: Demand bars for 0–30 and 30–60 min per hour
- `1hour`: One value per hour
- `2hour`: One value every 2 hours

Note:
- If the recording interval doesn’t align with the demand window, demand is calculated for the **nearest aligned window**.
- Example: If interval = 30min starting at 00:10, demand values shift 10 min forward from standard hour marks.

Example:  
- `Dem P+` at 19:30–20:00 = 213.2 kW  
- Highest bar = 374.1 kW

![demand-tab-example](assets/demand-tab-example.jpg)

#### ■ [mains signaling] Tab

The **mains signaling tab** displays trend graphs for voltage parameters related to **mains signaling**, which is used to remotely control industrial equipment.

##### What is mains signaling?

Mains signaling is a control signal defined by **IEC 61000-4-30**.  
It is one of the required measurement parameters for power quality monitoring in systems where **control signals are transmitted via voltage** through the mains supply.

This signal is often used to:
- Control street lighting, heating/cooling, or factory automation
- Provide off-peak operation signals
- Trigger load-shedding devices

> For full signal behavior and handling, refer to PQ3198 instruction manual (V2.00 or later)


##### Applicable Models and Parameters

This tab is only available on **PQ3198 (V2.00 or later)**.

Selectable parameters:

- `Msv1`: Signaling voltage level 1
- `Msv1%`: Signaling content percentage 1
- `Msv2`: Signaling voltage level 2
- `Msv2%`: Signaling content percentage 2

- `Msv1`, `Msv2` are absolute voltage values
- `Msv1%`, `Msv2%` are expressed as percentage of nominal voltage

##### Example Readings:

On a given time window:
- `Msv1 AVG`: 0.354 V  
- `Msv2 AVG`: 0.533 V  
- `Msv1% AVG`: 0.45%  
- `Msv2% AVG`: 0.35%

This tab allows real-time monitoring of signaling performance and interference analysis.

![mains-signaling-tab](assets/mains-signaling-tab.jpg)

## 3.4 Reviewing events

Using the event list on the left side of the window, you can check the time and characteristics of any events that occurred.

### Event Statistics Graph

![event-statistics-graph](assets/event-statistics-graph.jpg)

- The Event Statistics Graph displays the number of events that occurred as a bar graph.
- Events included in the statistics graph can be reviewed on trend graphs.
- ▼ indicates the selected event on the event list.
- ▽ indicates the event's start time.

#### Three key settings:

1. **Event filter**:
   - `[All]`: include all events in the count.
   - `[Custom]`: select multiple specific event types.
   - Only IN time is counted for events with IN/OUT timestamps.

2. **Type of statistics**:
   - `[View by date]`: tabulates number of events per day.
   - `[View by hour]`: tabulates number of events per hour.
   - `[ITIC Curve]`: displays tolerance curve (→ p.40).
   - `[User-defined Curve]`: displays custom curve (→ p.41).

3. **Statistic period**:
   - Set start and end dates to tally or visualize statistics.

Examples:

![event-statistics-view-by-date](assets/event-statistics-view-by-date.jpg)  
![event-statistics-view-by-hour](assets/event-statistics-view-by-hour.jpg)

---

### Event List

![event-list-cursor-mode](assets/event-list-cursor-mode.jpg)
- Shows IN time, OUT time, worst value, and level for the selected event.

- Clicking on the bar graph selects corresponding data in the event list.
- When the cursor is hidden, all events are displayed.
- Events shown in the statistics graph are bolded.
- If the distribution graph is deselected, the list shows **all** events in that time period.
- Clicking `[+]` expands grouped events.

#### Fields in Event List:

- **Time**: timestamp of the event.
- **Event**: event type (red font indicates special event).
- **I/O**: IN = start, OUT = end.
- **CH**: affected channel.
- **Duration**: duration from IN to OUT.
- **Worst**: worst value observed during the event.


#### Notes:

- For PW3198:
  - `Irms1/2` is shown as `Inrush`.
- For PW3198 / PQ3198:
  - `Cont` is shown as `After`.
- For `After` (Cont) events:
  - If end time of waveform ≠ next start time, it may not be displayed.

---

## ITIC Curve

![itic-curve-overview](assets/itic-curve-overview.jpg)

The ITIC curve plots voltage swell, dip, and interruptions over time.

- X-axis: Elapsed Time [s]
- Y-axis: % of Nominal Voltage

#### ITIC Event Table

- **Voltage Swell**:
  - Time: Continuous Swell Duration
  - Value: Maximum Swell Voltage

- **Voltage Dip**:
  - Time: Dip Duration
  - Value: Residual Voltage (Dip Voltage Minimum)

- **Interruption**:
  - Time: Continuous Interruption Duration
  - Value: Residual Voltage (Interruption Voltage Minimum)

- **Transient**:
  - Time: Transient Width
  - Value: Maximum Transient Voltage

#### Other details:

- Black lines = upper and lower tolerance curves.
- Total events and violations (upper/lower) are displayed at the top.
- Event markers:
  - CH1: Red ◆
  - CH2: Green ◆
  - CH3: Blue ◆
  - ◆ = selected, ◇ = unselected

> *Events shorter than 1μs will not be shown.*

---

## User-defined Curve

![user-defined-curve-overview](assets/user-defined-curve-overview.jpg)

You can customize the tolerance curve shown on the graph.

- Access via right-click → `Tolerance curve setting`
- Default values follow ITIC curve

Settings include:

- **Event Type**:
  - Options: Tran / Swell / Dip / Intrpt
- **YDiv Maximum**: 500% to 2000% (default 100%)
- **TDiv Minimum**: 1μs, 100μs, 1ms

![user-defined-settings](assets/user-defined-settings.jpg)

You can directly edit the upper and lower curve tables:
- Custom time-voltage coordinate pairs can be input.
- You may use `u` instead of the unit `μ`.


## 3.5 Viewing event data