from classes import *


def clicked(b):
    temp = b.tooltip
    if b.value == True:
        b.tooltip = b.description
        b.description = temp
        b.value = False
    else:
        b.tooltip = b.description
        b.description = temp
        b.value = True


def deleteFiles(b):
    for folder in os.listdir(offline_plots):
        for the_file in os.listdir(os.path.join(offline_plots, folder)):
            file_path = os.path.join(offline_plots, folder, the_file)
            if os.path.isfile(file_path):
                os.remove(file_path)


button_exec = widgets.Button(description=turn_on_exec, tooltip=turn_off_exec, value=True)
button_exec.on_click(clicked)

button_plots = widgets.Button(description=turn_off_plots, tooltip=turn_on_plots, value=True)
button_plots.on_click(clicked)

button_delete = widgets.Button(description=delete_files, value=True)
button_delete.on_click(deleteFiles)

display(button_exec)
display(button_plots)
display(button_delete)
