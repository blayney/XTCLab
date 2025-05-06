from spatialaudiometrics import load_data as ld
from spatialaudiometrics import visualisation as vis
import matplotlib.pyplot as plt  # Ensure matplotlib is imported

hrtf = ld.HRTF("P0275_Windowed_96kHz.sofa")

fig,gs = vis.create_fig()

axes = fig.add_subplot(gs[1:6,1:6])

vis.plot_itd_overview(hrtf)
vis.plot_ild_overview(hrtf)

vis.plot_hrir_both_ears(hrtf,45,0,axes)
axes = fig.add_subplot(gs[1:6,7:12])
vis.plot_hrtf_both_ears(hrtf,45,0,axes)

axes = fig.add_subplot(gs[7:12,1:6])
vis.plot_hrir_both_ears(hrtf,0,-30,axes)
axes = fig.add_subplot(gs[7:12,7:12])
vis.plot_hrtf_both_ears(hrtf,0,-30,axes)

vis.show()
plt.show()  # Explicitly call plt.show() to keep the plot open
