This is a toolkit for loading, creating and tuning HRTFs for crosstalk cancellation. It includes a library, xtc.py, that handles loading, analysis and creation of SOFA files (including test code that verifies relevant geometry exists in a given SOFA file, for example two virtual sources at [-30, 30] degree azimuth to match your setup). The library also generates graphs allowing you to analyse SOFA files' HRIRs, including producing SMAART style phase and transfer function plots for any given azimuth angle in the dataset available.

It also includes a Qt based simulator for demonstrating XTC, creating basic virtual HRTFs (with export to SOFA) and automatically generating XTC filters for these virtual HRTFs to demonstrate "ideal" crosstalk cancellation filters. It shows transfer functions for all four channels (both TFs for contralateral and ipsilateral pairs are calculated), and has an interactive 2D rendering allowing the virtual listener head to be moved around in space, so that you can visualize the deteriation of the XTC effect as the listener moves away from the "sweet spot".

You can also recalculate new virtual HRTFs and XTC filters with the head anywhere in the space, adjust filter regularization and view the impact on transfer functions in real time.

Finally, the simulator includes an energy distribution calculator, which computes the relative contralateral energy reduction for each point in the 4m x 4m virtual space around the ideal position, showing how the filter's resulting interference pattern radiates out from the speakers, and develops as you move away from the sweet spot. It's also interesting to see how the regularisation factor impacts this interference pattern.

Here's a screenshot of the most up to date version
![Screenshot 2025-05-06 at 13 03 36](https://github.com/user-attachments/assets/e10810bc-698d-4952-926a-847c00873da8)

<img width="1450" alt="Screenshot 2025-03-31 at 15 43 36" src="https://github.com/user-attachments/assets/21f6d868-ba4d-4f03-b778-10f10a984479" />
<img width="1450" alt="Screenshot 2025-01-14 at 17 06 15" src="https://github.com/user-attachments/assets/4dd0ba8a-4d6c-474b-b1a0-6d53c5aeabe3" />
