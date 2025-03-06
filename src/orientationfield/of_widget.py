"""OrientationField widget module.
"""
from qtpy.QtCore import Qt
import napari._qt.layer_controls.qt_colormap_combobox
import napari._qt.layer_controls.qt_image_controls_base
import napari._qt.layer_controls.qt_shapes_controls
import napari._qt.widgets.qt_color_swatch
import napari._qt.widgets.qt_theme_sample
from qtpy.QtGui import QColor, QMouseEvent
from napari.layers import Image, Points # for magicgui and selection error handling
from magicgui import magicgui
import pathlib
from qtpy.QtWidgets import QFileDialog, QWidget
from qtpy import uic
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from .of_script import compute_nematic_field, draw_nematic_field_svg, find_defects, extract_nematic_points_layer




def embed(wid):
    """Create GUI elements specific to a widget. 
    Can't be a method because _do_all relies on the magicgui decorator, 
    which doesn't work well inside class definitions.
    """
    @magicgui(
        call_button=False,
        image={'label':'Image'}
    )
    def _do_all(
        image:Image
    ):
        colormap = wid.colormapselect.currentData()
        if wid.colortype.currentData() == "fixed":
            custom_kwargs = f"color:{tuple(wid.colorswatch._color)}"
        else:
            custom_kwargs = f"color:{wid.colortype.currentData()}-colormap:{colormap}"
        # edge width ?
        
        img = wid.imgCombobox.currentData()
        success = compute_nematic_field(img, sigma=wid.sigmaSpin.value())
        if not success:
            return False
        box_size = wid.boxsizeSpin.value()
        custom_kwargs += f"-edge_width:{max(1,int(box_size/10))}" # edge width quasi-fixed ratio of box_size
        thresh = wid.threshSpin.value()
        magnitude_as_length = wid.lengthsCheckbox.isChecked()
        length_scale = wid.lengthsSpin.value()
        mode = wid.defectsCombo.currentData()
        draw_nematic_field_svg(
            img=img,
            box_size=box_size,
            thresh=thresh,
            color=True,
            lengths=magnitude_as_length,
            length_scale=length_scale,
            custom_kwargs=custom_kwargs
        )
        if wid.defectsCheckbox.isChecked():
            find_defects(
                img=img,
                box_size=box_size,
                thresh=thresh,
                mode=mode
            )
        return True

    def _save_as_csv():
        file = QFileDialog.getSaveFileName(filter="napari builtin points (*.csv)")
        if file[0] == '': return 
        img = wid.imgCombobox.currentData()
        box_size = wid.boxsizeSpin.value()
        name = file[0] if file[0][-4:] == '.csv' else file[0]+".csv"    
        points:Points = extract_nematic_points_layer(img, box_size=box_size, return_early=True)
        points.save(name)
    
    return _do_all, _save_as_csv
    



class DoAllWidget(QWidget):
    """OrientationField Widget. Contains all main features of the of_script scripting module."""

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        uic.loadUi(pathlib.Path(__file__).parent / "./wid.ui", self)

        _do_all, _save_as_csv = embed(self)

        img_selector = self.viewer.window.add_dock_widget(_do_all)
        self.imgCombobox = img_selector.children()[4].children()[1].children()[2]
        self.rightLayout.insertWidget(0, self.imgCombobox)
        self.viewer.window.remove_dock_widget(img_selector)


        # CUSTOM COLOR WIDGETS --
        self.colormapselect = napari._qt.layer_controls.qt_colormap_combobox.QtColormapComboBox(None)
        self.rightLayout.insertWidget(10,self.colormapselect)
        for name, cm in napari._qt.layer_controls.qt_image_controls_base.AVAILABLE_COLORMAPS.items():
            self.colormapselect.addItem(cm._display_name, name)

        def mouseReleaseEvent(obj, event: QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                initial = QColor(*(255 * obj._color).astype('int'))
                popup = napari._qt.widgets.qt_color_swatch.QColorPopup(obj, initial)
                popup.colorSelected.connect(obj.setColor)
                popup.show()
        colorpicker = napari._qt.layer_controls.qt_shapes_controls.QColorSwatchEdit(None, initial_color=[0,0,1,1])
        self.rightLayout.insertWidget(10,colorpicker)
        self.colorswatch = colorpicker.children()[-1]
        self.colorswatch.mouseReleaseEvent = lambda event : mouseReleaseEvent(self.colorswatch, event)
        self.colormapselect.setVisible(False)
        colorpicker.setVisible(True)

        def _change_colorselecter(text):
            if text == 'fixed': 
                self.colormapselect.setVisible(False)
                colorpicker.setVisible(True)
            else: 
                self.colormapselect.setVisible(True)
                colorpicker.setVisible(False)

        self.colortype = napari._qt.widgets.qt_theme_sample.QComboBox(None)
        for name, dname in zip(["fixed", "angle", "norm"], ["fixed", "angle", "norm"]):
            self.colortype.addItem(dname,name)
        self.colortype.currentTextChanged.connect(_change_colorselecter)
        self.colortype.adjustSize()
        self.rightLayout.insertWidget(8,self.colortype)

        # CUSTOM SAVE BUTTON --
        save_as = napari._qt.widgets.qt_theme_sample.QPushButton()
        self.layout().addWidget(save_as)
        save_as.setText('Save nematics as CSV...')
        save_as.clicked.connect(_save_as_csv)
        save_as.move(10,345)
        save_as.resize(120,20)

        fig, ax = plt.subplots(figsize=(2, 0.5), layout='constrained')
        fig.patch.set_facecolor('#00000000')
        self.defectsColormap = self.bottomLayout.insertWidget(1,mpl.backends.backend_qtagg.FigureCanvas(fig))
        cmap = ListedColormap(
            ['#00ff30', '#00ddd1', '#6666ff', '#999999', '#ff6666', '#dd7600', '#ffd700']
        )
        norm = mpl.colors.Normalize(vmin=-1.75, vmax=1.75)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        cbar.outline.set_edgecolor("#00000000")
        cbar.ax.xaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color="white")
        
        

        self.lengthsCheckbox.setChecked(True)

        self.computeButton.clicked.connect(_do_all)
        for name, dname in zip(["squares", "simplified"], ["squares", "simplified"]):
            self.defectsCombo.addItem(dname,name)
        
        # TOOLTIPS
        self.imgCombobox.setToolTip("The selected Image layer.")
        self.sigmaSpin.setToolTip("Bandwidth of kernel : range of orientation computation.")
        self.threshSpin.setToolTip("Nematic magnitude below which nematic tensors aren't considered.")
        self.defectsCombo.setToolTip("Defects drawing style.")
        self.defectsCheckbox.setToolTip("Draw defects.")
        self.boxsizeSpin.setToolTip("Size of boxes over which the nematic field is averaged.")
        self.lengthsCheckbox.setToolTip("Vary drawn bars' lengths according to magnitude.")
        self.lengthsSpin.setToolTip("Bars lengths scaling factor.")
        save_as.setToolTip("Save nematic field as CSV.")
        self.colortype.setToolTip("Color according to...")
        colorpicker.setToolTip("Chosen fixed color.")
        self.colormapselect.setToolTip(f"Chosen colormap .")
        

