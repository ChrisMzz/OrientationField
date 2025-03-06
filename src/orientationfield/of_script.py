"""OrientationField scripting module. Many functions in here refer to the exact term, that is "nematic field".
"""
from typing import overload
from napari import Viewer
from qtpy.QtCore import Qt
import napari._qt.layer_controls.qt_colormap_combobox
import napari._qt.layer_controls.qt_image_controls_base
import napari._qt.layer_controls.qt_shapes_controls
import napari._qt.widgets.qt_color_swatch
import napari._qt.widgets.qt_theme_sample
import warnings
from qtpy.QtGui import QColor, QMouseEvent
from napari.layers import Image, Points, Labels # for magicgui and selection error handling
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import pathlib
import numpy as np
from qtpy.QtWidgets import QFileDialog, QWidget, QMessageBox
from qtpy import uic
from skimage import io, measure
import orientationfield.nematicfield as nf
import sys, os
import re
from magicgui.tqdm import tqdm

viewer = napari.current_viewer()

# wrote docstrings to make use of https://github.com/pyapp-kit/magicgui/issues/107
# PR #100 from napari/magicgui linked in there refers to this feature added after https://github.com/pyapp-kit/magicgui/issues/383

def nematic_field_properties():
    """Boilerplate code to make properties of a nematic field Layer."""
    return {"Qxx":[], "Qxy":[], "norm":[], "angle":[]}

def hex_to_rgba(h):
    if h[0] == '#': h = h[1:]
    if len(h) == 6: h += 'ff'
    return np.array([int(d+u,16) for d,u in zip(list(h[::2]),list(h[1::2]))])

def proj_distance(p:tuple, l:tuple[tuple]):
    """Distance between a point `p` and its projection on the extension of a line passing by two points, provided in a tuple `l`.
    Also known as the "perpendicular distance"."""
    (l1x, l1y), (l2x, l2y) = l
    length = np.sqrt((l2x-l1x)**2 + (l2y-l1y)**2)
    if length == 0: return np.sqrt((l1x-p[0])**2 + (l1y-p[1])**2)
    return np.abs( (l2y - l1y)*p[0] - (l2x - l1x)*p[1] - l2y*l1x + l2x*l1y ) / length

def rdp(points_list, eps):
    """Internal, used by `rdp_polygon`."""
    dmax, index = 0, 0
    for i in range(1,len(points_list)-1):
        d = proj_distance(points_list[i],(points_list[0],points_list[-1]))
        if d > dmax: index, dmax = i, d
    return np.array([*rdp(points_list[:index],eps), *rdp(points_list[index:],eps)]) if dmax > eps else [points_list[0], points_list[-1]]
    
def rdp_polygon(polygon, eps):
    """Ramer-Douglas-Peucker algorithm for vertex reduction. Adapted to work on a polygon.
    Splits the polygon into two lines based on the "middle" vertex, and performs the classical algorithm
    as described in [the wikipedia page](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm#Pseudocode).

    Args:
        polygon (list[np.ndarray]): List of points used to construct the polygon.
        eps (float): Minimal distance between two vertices in the result.
    """
    if eps == 1: return polygon
    return np.concatenate([rdp(polygon[:len(polygon)//2], eps), rdp(polygon[len(polygon)//2:], eps)[:-1]])




@magicgui(
    call_button='Compute Nematic Field',
    img={'label':'Image'},
    sigma={'value':3.0},
    cutoff_ratio={'value':2.0, 'visible':False},
    normalize_options={'choices':['total','per frame']}
)
def compute_nematic_field(
    img:Image, 
    sigma:float=3.0, 
    cutoff_ratio:float=2.0, 
    decolorize:bool=False, 
    decolorize_axis:int=2, 
    normalize:bool=True,
    normalize_options:str='total'
    ):
    """Compute nematic field of an Image layer `img`, and stores the result in layer metadata as a `img.data.shape+(2,2)`-array keyword "nematic_field".

    Args:
        img (Image): the Image layer.
        
        sigma (float, optional): Bandwidth of kernel. Defaults to 3.0.
                
        decolorize (bool, optional): Whether image has an RGB (or RGBA) channel that needs to be collapsed. Defaults to True.
        
        decolorize_axis (int, optional): The axis of the image that has the RGB (or RGBA) channel. Defaults to 2.
        
        normalize (bool, optional): Whether to normalize pixel intensity between 0 and 1. Defaults to True.
        
        normalize_options (str, optional): Type of normalization. 'total' normalization normalizes with respect to the whole image, all channels included (time included) 'per frame' normalizes each slice along the first axis individually. Defaults to 'total'.
    
    Example:
        >>> viewer.add_image(img_array) # img_array is a numpy array
        >>> img_layer = viewer.layers[-1] # get the corresponding Image layer
        >>> nf_script.compute_nematic_field(img_layer, sigma=3.0, cutoff_ratio=2.0, decolorize=False)
        >>> nem_field = img_layer.metadata["nematic_field"]
        
    """
    if len(img.data.shape) > 2 and img.data.shape[-1] in [3,4]:
        # if 3 or 4 is the number of images in your image sequence, please transpose your image so that 
        # time along the first axis
        if viewer is not None:
            instructions = QMessageBox()
            instructions.setText("You seem to be using the nematic field computation on an RGB or RGBA image. Please convert to grayscale before using the tool.")
            instructions.setWindowTitle('Warning')
            instructions.exec()
        else:
            raise ValueError("You seem to be using the nematic field computation on an RGB or RGBA image. Please convert to grayscale before using the tool.")
        return False
    if decolorize: img_data = np.array(np.mean(img.data,axis=decolorize_axis), dtype=np.float64) # RGBA channel
    else: img_data = img.data
    if normalize_options == 'total':
        normalized_img = (img_data-np.min(img_data))/(np.max(img_data)-np.min(img_data))
    elif normalize_options == 'per frame':
        normalized_img = np.array([(frame-np.min(frame))/(np.max(frame)-np.min(frame)) for frame in img_data])
    else: normalized_img = img_data
    if len(img_data.shape) > 2:
        nem_field = np.array([nf.nematic_field(frame, sigma=sigma, cutoff_ratio=cutoff_ratio) for frame in tqdm(normalized_img if normalize else img_data, desc='Computing...', leave=None)])
    else: nem_field = nf.nematic_field(normalized_img if normalize else img_data, sigma=sigma, cutoff_ratio=cutoff_ratio)
    img.metadata["nematic_field"] = nem_field
    return True

@magicgui(
    call_button='Preview Kernel',
)
def preview_kernel(
    **kwargs
    ) -> np.ndarray: # uses current values in GUI
    """Display and/or return kernels for a given bandwidth and cutoff ratio. Unless specified, uses values in `compute_nematic_field` GUI.
    
    Kwargs:
        sigma (float): Bandwidth of the kernel.
        cutoff_ratio (float): Ratio between size of kernel and bandwidth.
        show (bool): Whether to add the kernel image to the viewer or not. Defaults to True.

    Returns:
        np.ndarray: Kernel (Kxx as first component and Kxy as second component).
    """
    sigma = compute_nematic_field.__signature__.parameters['sigma'].default if 'sigma' not in kwargs.keys() else kwargs['sigma']
    cutoff_ratio = compute_nematic_field.__signature__.parameters['cutoff_ratio'].default if 'cutoff_ratio' not in kwargs.keys() else kwargs['cutoff_ratio']
    kernels = np.array(nf.kernels(sigma, cutoff_ratio))
    if 'show' in kwargs.keys():
        if not kwargs['show']: return kernels
    viewer.add_image(kernels, name=f'kernels - sigma {sigma} / cutoff_ratio {cutoff_ratio}')
    return kernels
    


@magicgui(
    call_button='Draw Nematic Points Layer',
    img={'label':'Image'},
    mask={'label':'Mask'},
    box_size={'value':8},
    folder={"label": "Folder", "mode":"d"},
    return_early={'visible':False}
)
def extract_nematic_points_layer(
    img:Image,
    mask:Labels|None,
    folder:pathlib.Path,
    box_size:int=8,
    invert_mask:bool=False,
    return_early:bool=False
    ):
    """Generate a points layer for an Image layer, to save to a csv in a specified folder.
    Points will be positioned in the center of boxes as they would be drawn in `draw_nematic_field`.
    To manually set the box size (in case of exterior use of this function), use the `box_size` keyword argument.
    
    For video/multichannel image layers, this function will generate a video of points layers (which can be split later
    for batch processing).

    If nematic field for the image is not computed, it will be generated with default parameters.

    Args:
        img (Image): Image layer.
        mask (Labels): Labels of mask used to filter which points to include. Optional.
        folder (pathlib.Path): path to folder where the CSV will be saved.
        box_size (int): Size of the boxes whose centers will be the points of this layer.
        invert_mask (bool): Whether mask filters in or filters out parts of the image.
        
    Returns:
        Points layer (for scripting) if return_early is True. Otherwise, returns None.
        
    Example:
        >>> viewer.add_image(img_array) # img_array is a (h,w) array
        >>> img_layer = viewer.layers[-1] # get the corresponding Image layer
        >>> of_script.compute_nematic_field(img_layer, sigma=3.0, cutoff_ratio=2.0, decolorize=False)
        >>> points = of_script.extract_nematic_points_layer(img_layer, box_size=8, return_early=True)
    """

    #try:     param_names = [param.name for param in params] 
    #except : param_names = [param for param in params]
    param_names = ["Qxx", "Qxy", "norm", "angle"]
    # convert shapes to labels
    if "nematic_field" not in img.metadata.keys(): 
        success = compute_nematic_field(img)
        if not success:
            return False
    nem_field = img.metadata["nematic_field"]
    if mask is None: mask_data = np.ones(nem_field.shape[:-2])
    else: mask_data = mask.data
    if len(mask_data.shape) != len(nem_field.shape[:-2]): mask_data = np.array([mask_data for _ in range(nem_field.shape[0])])
    name = img.name
    if len(nem_field.shape) > 4: h,w = nem_field.data.shape[1:3]
    else: h,w = nem_field.data.shape[:2] # image shape    
    x = y = box_size
    h_overflow, w_overflow = (h%x), (w%y)
    t_crop, b_crop, l_crop, r_crop = 0,-h,0,-w
    if h_overflow + w_overflow != 0:
        t_crop = h_overflow//2; b_crop = h_overflow-t_crop
        l_crop = w_overflow//2; r_crop = w_overflow-l_crop
    total_points, total_properties = nf.extract_points(nem_field, mask_data, box_size, param_names, invert_mask)
    metadata = {"shape":nem_field.shape[:-2], "box_size":box_size, "offset":(t_crop, l_crop), "name":img.name}
    points = viewer.add_points(np.array(total_points), properties=total_properties, metadata=metadata, size=2, name=f'{name} - nematic points layer')
    if return_early: return points # for use in defects
    if points.data.shape[1] < 3: points.save(os.path.join(str(folder),f'{name}.csv')); del viewer.layers[viewer.layers.index(points)]; return
    if points.data.shape[1] != 3: raise Exception(f"Can't handle shape of point data {points.data.shape}."); return
    if not os.path.isdir(os.path.join(str(folder),name)): os.mkdir(os.path.join(str(folder),name))
    time_length = np.max(points.data,axis=0)[0]+1
    properties_names = list(points.properties.keys())
    for t in tqdm(range(time_length)):
        new_points = points.data[points.data.transpose(1,0)[0]==t].transpose(1,0)[1:3].transpose(1,0)
        new_properties = {}
        for property_name in properties_names:
            new_properties[property_name] = points.properties[property_name][points.data.transpose(1,0)[0]==t]
        new_layer = viewer.add_points(np.array(new_points), properties=new_properties, name=f'{name} - t{t}')
        new_layer.save(os.path.join(str(folder),name,f'{name} - t{t}.csv')) # save to pathlib folder with its current default name
        del viewer.layers[-1]
    del viewer.layers[viewer.layers.index(points)]
    return True


@magicgui(
    call_button='Draw Nematic Field SVG',
    img={'label':'Image'},
    box_size={'value':8},
    length_scale={'value':1.5, 'min':1e-2, 'max':10, 'step':1e-2},
    thresh={'value':1e-4, 'min':1e-12, 'max':1, 'step':1e-8},
    ordered={'visible':False}
)
def draw_nematic_field_svg(
    img:Image,
    box_size:int=8,
    thresh=1e-4,
    color:bool=True,
    lengths:bool=False,
    length_scale:float=1.5,
    custom_kwargs:str='',
    ordered:int=0
):
    """Napari-specific function for generating a Shapes layer with vector representation of every nematic in the field.

    Args:
        img (Image): Image layer.
        box_size (int): Size of boxes over which the field is averaged.
        thresh (float): Nematic norm threshold. Norms are generally very low so this needs to be really low as well.
        color (bool): Whether to use the magma cmap for the nematic field or not.
        custom_kwargs (str): String of custom keyword arguments passed to `viewer.add_shapes` on layer generation. Keywords should be specified in the format 'arg:val' seperated by '-', without spaces. Ignores the color argument explained previously.
    
    Example:
        >>> nemfield = of_script.draw_nematic_field_svg(img, box_size=8, custom_kwargs:'edge_color:angle-edge_colormap:hsv')
    """
    #length_scale=1.5

    def _process_color(val):
        if val[0] == '#': return val
        if val[0] not in ['[', '(']: return val
        val = [float(v) for v in re.split(',', val[1:-1])]
        if np.max(val) > 1: 
            val = [int(v) for v in val]
            return val + [255] if len(val) == 3 else val
        else:
            return val    
    def _get_crop(h,w):
        h_overflow, w_overflow = (h%box_size), (w%box_size)
        t_crop, b_crop, l_crop, r_crop = 0,-h,0,-w
        if h_overflow + w_overflow != 0:
            t_crop = h_overflow//2; b_crop = h_overflow-t_crop
            l_crop = w_overflow//2; r_crop = w_overflow-l_crop
            if b_crop == 0: b_crop = -h
            if r_crop == 0: r_crop = -w
        return t_crop, b_crop, l_crop, r_crop
    def _make_lines(frame, t=None):
        boxes = nf.tesselate(frame[t_crop:-b_crop,l_crop:-r_crop], box_size, box_size)
        nems = []
        properties = {"Qxx":[], "Qxy":[], "norm":[], "angle":[]}
        for box, pos in boxes:
            nem = np.mean(box, (0,1))
            Qxx, Qxy = nem[0]
            Qnorm, phi = nf.extract(nem)
            if Qnorm < thresh: continue
            t1, t2 = (pos[0]+pos[1])//2 + t_crop, (pos[2]+pos[3])//2 + l_crop
            default_length = (box_size//2)*np.exp(1j*phi)
            if lengths:
                p1 = complex((t1+t2*1j) - 1j*default_length*length_scale*(Qnorm/maxnorm))
                p2 = complex((t1+t2*1j) + 1j*default_length*length_scale*(Qnorm/maxnorm))
            else:
                p1 = complex((t1+t2*1j) - 1j*default_length)
                p2 = complex((t1+t2*1j) + 1j*default_length)
            properties["Qxx"].append(Qxx), properties["Qxy"].append(Qxy)
            properties["norm"].append(Qnorm), properties["angle"].append(phi)
            if t == None:
                if ordered != 0: # meant for 3D view only. testing to see if dynamic slicing is achievable
                    warnings.warn("You are using the `ordered` parameter, meant for testing magnitude-slicing.\nThis parameter is not meant for practical use at the moment.")
                    if ordered == 1:
                        order_t = (properties["norm"][-1]/maxnorm)*2000
                    else:
                        order_t = (properties[ordered][-1])*100
                    nems.append(np.array([[order_t, p1.real, p1.imag],[order_t, p2.real, p2.imag]]))
                else:
                    nems.append(np.array([[p1.real, p1.imag],[p2.real, p2.imag]]))
            else:
                nems.append(np.array([[t, p1.real, p1.imag],[t, p2.real, p2.imag]]))
        return nems, properties

    name = img.name
    if "nematic_field" not in img.metadata.keys(): 
        success = compute_nematic_field(img) # uses current values in GUI
        if not success:
            return False
    img.metadata["box_size"] = box_size
    nem_field = img.metadata["nematic_field"]
    if len(nem_field.shape) > 4:
        h, w = nem_field.shape[1:3]
        t_crop, b_crop, l_crop, r_crop = _get_crop(h,w)
        total_nems = []
        maxnorm = np.max([[nf.extract(np.mean(box, (0,1)))[0] for box, pos in nf.tesselate(frame[t_crop:-b_crop,l_crop:-r_crop], box_size, box_size)] for frame in nem_field])
        total_properties = {"Qxx":[], "Qxy":[], "norm":[], "angle":[]}
        for t, frame in enumerate(nem_field):
            nems, properties = _make_lines(frame, t)
            for p in total_properties.keys(): total_properties[p] += properties[p] 
            total_nems += nems
    else:
        h, w = nem_field.shape[:2]
        t_crop, b_crop, l_crop, r_crop = _get_crop(h,w)
        boxes = nf.tesselate(nem_field[t_crop:-b_crop,l_crop:-r_crop], box_size, box_size)
        maxnorm = np.max([nf.extract(np.mean(box, (0,1)))[0] for box,pos in boxes])
        total_nems, total_properties = _make_lines(nem_field)     
    kwargs = {"edge_color":"black", "edge_width":1, "shape_type":'line', "opacity":1, "blending":"translucent"}   
    if color:
        kwargs["edge_color"] = "norm"
        kwargs["edge_colormap"] = "magma"
    if custom_kwargs != '':
        for argval in re.split('-', custom_kwargs):
            arg, val = re.split(':', argval)
            if arg in ['color', 'colormap', 'width', 'contrast_limits']: arg = 'edge_' + arg
            if arg in ['opacity','edge_width','z_index']: val = float(val)
            if arg == 'edge_color': val = _process_color(val)
            if arg == 'edge_contrast_limits': val = [float(v) for v in re.split(',', val[1:-1])]
            kwargs[arg] = val   
    return viewer.add_shapes(
        total_nems, properties=total_properties, name=f"{name} - nematic field",
        **kwargs
        )



@magicgui(
    call_button='Cluster defects',
    points={'label':'Points'},
    thresh={'visible':False},
    mode={'visible':False}
)
def cluster_defects(points:Points, thresh:float=-1, mode:str='simplified'):
    """Current (faulty) implementation for finding potential defects. Has to be fined tuned per image, which is very bad. 

    Args:
        points (Points): The Points layer, computed after masking (or not).

    Returns:
        Points: Points layer of defects (points in red).
    """
    rdp_eps = 1
    box_size = points.metadata["box_size"]
    t_crop, l_crop = points.metadata["offset"]
    if thresh == -1: thresh = draw_nematic_field_svg.__signature__.parameters["thresh"].default

    offset = -3 # has to be negative ; arbitrarily chosen

    def _find_og_position(p0,p1):
        return (p0*box_size+t_crop+box_size//2, p1*box_size+l_crop+box_size//2)

    def maxconvolution(img):
        res = np.zeros(img.shape, dtype=int)
        m = max(1,box_size//4)
        for p0,p1 in nf.loop_over_positions(img.shape[:2]):
            sub = img[max(0,p0-m):min(p0+m,img.shape[0])+1,max(0,p1-m):min(p1+m,img.shape[1])+1]
            rgba = np.array([np.max(sub[:,:,0]),np.max(sub[:,:,1]),np.max(sub[:,:,2]),np.max(sub[:,:,3])])
            res[p0,p1] = rgba
        return res

    name = points.metadata["name"]
    # add 1 for padding
    points_list_normalised = [(point[0]//box_size+1, point[1]//box_size+1) for point in points.data]
    points_list = [list(point) for point in points.data]

    h,w = points.metadata["shape"]
    img = np.zeros((h//box_size+2, w//box_size+2)) # add 2 for padding

    for e,point in enumerate(points_list_normalised):
        if points.properties["norm"][e] < thresh:
            img[point] = 1
    clusters = measure.label(np.array(img, dtype=np.int32))
    img_contour = np.zeros((h//box_size+2, w//box_size+2))
    img_edges_small = np.zeros((h//box_size+2, w//box_size+2))
    properties = {"value":[], "color":[]}

    props = []
    edge_props = []
    for k in range(1,np.max(clusters)+1):
        cluster = (clusters==k)*offset # arbitrary negative offset for edge detection and debugging
        # edge detection accounts for box padding
        if np.any(cluster[1::(h//box_size-1)]==offset) or np.any(cluster[:,1::(w//box_size-1)]==offset):
            edge_props.append(np.array([
                _find_og_position(*(p-1))
                for p in measure.find_contours(cluster, positive_orientation='low', fully_connected='low')[0]
            ]))
            img_edges_small += k*(clusters==k)
            clusters -= k*(clusters==k)
            continue
        angles = []
        contour = []
        for e,p in enumerate(measure.find_contours(cluster, positive_orientation='low', fully_connected='low')[0]):
            contour.append(_find_og_position(*(p-1)))
            # find out on which side of the boundary the cluster is (accounts for padding)
            p0low, p1low = int(p[0]-1), int(p[1]-1)
            p0high, p1high = int(p[0]-.5), int(p[1]-.5)
            if cluster[p0low, p1low] == offset:
                cluster[p0high,p1high] = e+1
                r,c = _find_og_position(p0high,p1high)
            else:
                cluster[p0low,p1low] = e+1
                r,c = _find_og_position(p0low,p1low)
            if [r,c] not in points_list: break
            angles.append(points.properties["angle"][points_list.index([r,c])])
        else: # only reached if all [r,c] positions are valid boxes
            props.append(np.array(contour))
            img_contour += cluster

            diff = np.vectorize(np.subtract)(
                    angles,
                    [angles[-1]]+angles[:-1])
            diff -= np.pi*np.floor(diff/np.pi+0.5)
            value = np.sum(diff)/(2*np.pi)

            reached_target = False
            for target, color, charge in zip([-1.5,-1,-0.5,0,0.5,1,1.5], 
                ['#00ff30', '#00ddd1', '#6666ff', '#999999', '#ff6666', '#dd7600', '#ffd700'],
                ['-3/2', '-1', '-1/2', '', '1/2', '1', '3/2']):
                if (value-target)**2 < 0.01: 
                    defect_charge = charge
                    defect_color = color
                    reached_target = True
                    break
            if not reached_target: defect_charge, defect_color = 'N/A', '#ff33ff'
            properties["value"].append(defect_charge) 
            properties["color"].append(defect_color) 

    clusters = measure.label(np.array(clusters, dtype=np.int32))
    clusters_img = np.zeros((h,w,4), dtype=np.int32)
    edges_img = np.zeros((h,w,4), dtype=np.int32)
    for (p0,p1) in nf.loop_over_positions((h//box_size, w//box_size)):
        r,c = _find_og_position(p0,p1)
        clusters_img[r-box_size//2:r+(box_size+1)//2,c-box_size//2:c+(box_size+1)//2] = np.zeros(4) if clusters[p0,p1]-1 < 0 else hex_to_rgba(properties["color"][clusters[p0,p1]-1])
        edges_img[r-box_size//2:r+(box_size+1)//2,c-box_size//2:c+(box_size+1)//2] = np.ones(4)*(img_edges_small[p0,p1]>0) # hex_to_rgba here also ?
    
    if mode == "squares": 
        viewer.add_image(np.array(clusters_img, dtype=np.uint8), blending='translucent', opacity=0.7, name=f"{name} - clusters")
        viewer.add_image(np.array(edges_img, dtype=np.uint8), blending='translucent', opacity=0.7, name=f"{name} - edge clusters")
        # transtyping to uint8 because SVG export crashes otherwise
    
    # clean up edge_props
    edge_props = [edge_props[j] for j in [k for k,prop in enumerate([p if len(p)>3 else None for p in edge_props]) if prop is not None]]

    clusters_shapes = viewer.add_shapes(
        [rdp_polygon(prop[:-1],rdp_eps) for prop in props], # replace props by props[:3] and it does work, somehow
        shape_type='polygon', face_color=properties['color'], properties=properties,
        text={'string': '{value}', 'anchor': 'center', 'size': 8 }, edge_width=0, name=f"{name} - clusters")
    edge_clusters_shapes = viewer.add_shapes(
        [rdp_polygon(prop[:-1], rdp_eps) for prop in edge_props], shape_type='polygon', edge_width=0, name=f"{name} - edge clusters") 
    
    return clusters_shapes, edge_clusters_shapes


@magicgui(
    call_button='Box Intersection Defects',
    points={'label':'Points'}
)
def box_intersection_defects(points:Points):
    name = points.name
    box_size = points.metadata["box_size"]
    neighbours = lambda r,c : [[r+box_size,c], [r+box_size,c+box_size], [r,c+box_size]]
    new_defects = []
    properties = {"value":[], "color":[]}
    points_list = [list(point) for point in points.data]
    for idx, (r,c) in enumerate(points_list):
        if np.all([neighbour in points_list for neighbour in neighbours(r,c)]):
            idc = [idx] + [points_list.index(neighbour) for neighbour in neighbours(r,c)]
            angles = [points.properties["angle"][i] for i in idc]
            diff = np.vectorize(np.subtract)(
                    angles,
                    [angles[-1]]+angles[:-1])
            diff -= np.pi*np.floor(diff/np.pi+0.5)
            value = np.sum(diff)/(2*np.pi)

            if value**2 >= 0.01: # make sure defect charge is non zero
                new_defects.append(np.array([
                    [r,c], [r+box_size,c], [r+box_size,c+box_size], [r,c+box_size]
                ]))
            else:
                continue
            reached_target = False
            for target, color, charge in zip([-1,-0.5,0.5,1], 
                ['#00ddd1', '#6666ff', '#ff6666', '#dd7600'],
                ['-1', '-1/2', '1/2', '1']):
                if (value-target)**2 < 0.01: 
                    defect_charge = charge
                    defect_color = color
                    reached_target = True
                    break
            if not (-1.5 < value < 1.5):
                defect_charge = f'{int(value*2)}/2'
                reached_target = (value - int(value*2)/2)**2 < 0.01
                if value <= -1.5: defect_color = '#00ddd1'
                if value >= 1.5: defect_color = '#dd7600'
            if not reached_target: 
                defect_charge, defect_color = 'N/A', '#ff33ff'
                raise BaseException("Computation error, contact the developers with steps to reproduce.")
            properties["value"].append(defect_charge)
            properties["color"].append(defect_color)

    return new_defects, properties


@magicgui(
    call_button='Find defects',
    img={'label':'Image'},
    box_size={'visible':False},
    thresh={'visible':False},
    mode={'visible':False}
)
def find_defects(
    img:Image,
    box_size:int=-1,
    thresh:float=-1,
    mode:str='simplified'
):
    if len(img.data.shape) > 2: 
        warnings.warn("Defect detection can't be automated on image sequences. Please compute defects for each frame separately, using Split Stack or a separate script.")
        return 
    if thresh == -1: thresh = draw_nematic_field_svg.__signature__.parameters["thresh"].default
    if box_size == -1: box_size = draw_nematic_field_svg.__signature__.parameters["box_size"].default
    points = extract_nematic_points_layer(img, box_size=box_size, return_early=True)
    
    clusters, edge_clusters = cluster_defects(points, thresh, mode=mode)
    del viewer.layers[viewer.layers.index(points)]

    both = viewer.add_shapes(list(clusters.data) + list(edge_clusters.data), shape_type='polygon', name=f"{img.name} temp defects layer")

    points = extract_nematic_points_layer(img, mask=1*(both.to_labels(img.data.shape)==0), box_size=box_size, return_early=True)
    box_defects, box_properties = box_intersection_defects(points) # dependency on clusters ?
    if mode == "squares": 
        del viewer.layers[viewer.layers.index(clusters)]
        del viewer.layers[viewer.layers.index(edge_clusters)]
        clusters = viewer.add_shapes(
            box_defects, 
            properties=box_properties, 
            shape_type='ellipse', 
            face_color=box_properties["color"], 
            text={'string': '{value}', 'anchor': 'center', 'size': 8 },
            edge_width=0,
            name=f"{img.name} - points defects"
            )
    else:
        clusters.add_ellipses(box_defects, face_color=box_properties["color"])
        N = len(box_properties["value"])
        if N > 0:
            clusters.properties["value"][-N:] = box_properties["value"]
            clusters.properties["color"][-N:] = box_properties["color"]
    clusters.refresh_text()
    clusters.refresh_colors()
    clusters.refresh()
    clusters.text.visible = False
    
    del viewer.layers[viewer.layers.index(points)]
    del viewer.layers[viewer.layers.index(both)]



    



# ------------------------------------------


def get_help(widget):
    """Return tooltip of a widget. Currently the same as `widget.tooltip`.

    Args:
        widget (str or FunctionGui): The widget or widget name
    """

@overload
def get_help(widget:str):
    if widget not in tooltips.keys(): print(f"Couldn't find widget called {widget}."); return
    return tooltips[widget]

@overload
def get_help(widget:FunctionGui):
    return widget.tooltip


functions = [compute_nematic_field, preview_kernel, 
             extract_nematic_points_layer, 
             draw_nematic_field_svg, find_defects]
tooltips = {
    'compute_nematic_field':"""Compute nematic field of an Image layer `img`, and stores the result in layer metadata as a `img.data.shape+(2,2)`-array keyword "nematic_field".""",
    'preview_kernel':"""Display and/or return kernels for a given bandwidth and cutoff ratio. Unless specified, uses values in `compute_nematic_field` GUI.""",
    'extract_nematic_points_layer':"""Generate a points layer for an Image layer, to save to a csv in a specified folder.
    Points will be positioned in the center of boxes as they would be drawn in `draw_nematic_field`.
    To manually set the box size (in case of exterior use of this function), use the `box_size` keyword argument.
    
    For video/multichannel image layers, this function will generate a video of points layers (which can be split later
    for batch processing).

    If nematic field for the image is not computed, it will be generated with default parameters.""",
    'draw_nematic_field_svg':"""Napari-specific function for generating a Shapes layer with vector representation of every nematic in the field.""",
    'find_defects':""""""
}
# add docstrings for other functions



    



for function in functions: function._widget._mgui_set_tooltip(tooltips[function.name])


