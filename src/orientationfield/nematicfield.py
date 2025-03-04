import numpy as np
import skimage as sk
from scipy.signal import convolve
import sys






def loop_over_positions(shape, k=0) -> list[tuple]: 
    """Provides list of all positions in an ndarray of given shape.

    Args:
        shape (tuple): shape of ndarray
    """
    if k == len(shape)-1: 
        temp_list = []
        for j in range(shape[-1]): 
            temp = list(shape)
            temp[-1] = j
            temp_list.append(tuple(temp))
        return temp_list
    temp_list = []
    for j in range(shape[k]):
        temp = list(shape)
        temp[k] = j
        temp_list += loop_over_positions(tuple(temp), k+1)
    return temp_list

def make_random_line_image(h:int, w:int, n:int) -> np.ndarray: # eventually delete this function
    """Generate a mask of random lines for testing. Lines are of length somewhere in [0 to h/10] in height and somewhere in [0 to w/10] in width.

    Args:
        h (int): Height of image in px.
        w (int): Width of image in px.
        n (int): Maximum number of lines.

    Returns:
        ndarray: Array with values 0 and 1 representing mask of image.
    """
    img = np.zeros((h,w))
    for r0, c0 in zip(np.random.randint(0,h,n),np.random.randint(0,w,n)):
        dr,dc = np.random.randint(max(-int(h/10),-int(w/10)),min(int(h/10),int(w/10)),2)
        r1, c1 = r0+dr, c0+dc
        if (r0+dr >= h or c0+dc >= w) or (r0+dr < 0 or r0+dc < 0): continue
        rr, cc = sk.draw.line(r0,c0,r1,c1)
        img[rr,cc] = 1
    return img

def tesselate(img:np.ndarray, x:int, y:int) -> list[np.ndarray]:
    """Make list of rectangle subdivisions of an array, such that every element in original in array appears in one single subdivision.

    Args:
        img (np.ndarray): Initial array to subdivide.
        x (int): Width of box.
        y (int): Height of box.

    Raises:
        Exception: If provided th and tw aren't divisors of initial array, it's impossible to return such a subdivision array.

    Returns:
        list of ndarray and tuple representing each subdivision and the coordinates to their corners
    """
    h,w = img.shape[:2]
    th, tw = h//x, w//y # number of boxes along height, width
    if x*th != h or y*tw != w: raise Exception("Dimensions provided aren't divisors of image shape.")
    return [(img[dh:dh+x, dw:dw+y], (dh,dh+x, dw,dw+y)) for dw in range(0,w,y) for dh in range(0,h,x)]



def kernel(dr:np.ndarray, sigma:float) -> np.ndarray:
    """Kernel used in integration. Unused.
     
    Args:
        dr (np.ndarray): Vector along which the kernel is computed.
        sigma (float): bandwidth of the kernel.

    Returns:
        ndarray: (2,2)-array representing the resulting nematic.
    """
    if np.linalg.norm(dr) == 0: return np.zeros((2,2))
    phi = np.angle(1j*complex(*dr))
    phi += np.pi*(phi < 0) # angle in [0,pi)
    return np.exp(-(np.linalg.norm(dr)**2)/(2*sigma**2))*np.array([[np.cos(2*phi),  np.sin(2*phi)],
                                                                   [np.sin(2*phi), -np.cos(2*phi)]])

# kernels recently changed to scale down with sigma similarly to typical gaussians
def kernel_xx(dr:np.ndarray, sigma:float) -> float: # internal
    if sum(dr**2) == 0: return 0
    phi = np.angle(1j*complex(*dr))
    phi += np.pi*(phi < 0) # angle in [0,pi)
    return np.exp(-(sum(dr**2)**2)/(2*sigma**2))*np.cos(2*phi)/sigma

def kernel_xy(dr:np.ndarray, sigma:float) -> float: # internal
    if sum(dr**2) == 0: return 0
    phi = np.angle(1j*complex(*dr))
    phi += np.pi*(phi < 0) # angle in [0,pi)
    return np.exp(-(sum(dr**2)**2)/(2*sigma**2))*np.sin(2*phi)/sigma

def kernels(sigma:float, cutoff_ratio:float) -> tuple[np.ndarray]:
    """Returns array representing kernel of nematics over xx and over xy.

    Args:
        sigma (float): bandwidth of the kernels
        cutoff_ratio (float): radius/sigma ? might have to confirm with Matthias

    Returns:
        tuple[np.ndarray]: tuple of Kxx and Kxy kernels
    """
    radius = int(np.ceil(cutoff_ratio*sigma))
    if radius < 1: raise Exception('Please choose a bandwidth and cutoff_ratio such that the kernel radius is greater or equal to 3.')
    Kxx, Kxy = np.zeros((2*radius+1,2*radius+1)), np.zeros((2*radius+1,2*radius+1))
    #radius*np.linspace(-1,1,2*radius+1)
    center = np.array((radius, radius))
    for p in loop_over_positions(Kxx.shape): 
        if sum(np.array(p-center)**2) > radius**2: continue
        Kxx[p], Kxy[p] = kernel_xx(p-center, sigma), kernel_xy(p-center, sigma)
    return Kxx, Kxy
# return Kxx/(np.sum(Kxx>0)), Kxy/(np.sum(Kxy>0)) ?
    

def nematic_field(a:np.ndarray, sigma:float, cutoff_ratio:float) -> np.ndarray:
    """Run integration over a given area image array `a`. Each element in the array will be summed over and an "average nematic" 
    will be computed by integrating over a circle around it, of a radius defined by kernel bandwidth and cutoff ratio.

    Args:
        a (np.ndarray): area over which integral is calculated
        sigma (float): bandwidth of the kernel
        cutoff_ratio (float): radius/sigma ? might have to confirm with Matthias

    Returns:
        ndarray: (h,w,2,2)-array representing per-pixel nematics of the whole image.
    """
    Kxx, Kxy = kernels(sigma, cutoff_ratio)
    radius = int(np.ceil(cutoff_ratio*sigma))
    Qxx, Qxy = a*convolve(a-np.mean(a),Kxx, mode='full')[radius:-radius,radius:-radius], a*convolve(a-np.mean(a),Kxy, mode='full')[radius:-radius,radius:-radius]
    return np.array([[Qxx,Qxy],[Qxy,-Qxx]]).transpose(2,3,0,1)
# subtract mean of image to mitigate weighting on edges
# convolution on the whole image : done
# precomputed kernel as a separate function with cutoff and sigma (cutoff circle determines size) : done, but in loop
# averaging over the convolution with given box sizes - done, needs tweaking for parameter selection to be streamlined


def extract(Q:np.ndarray):
    """Get norm and angle from a (2,2)-array nematic.

    Args:
        Q (np.ndarray): (2,2)-array representing the nematic

    Returns:
        tuple: (norm, angle) where the angle is in [0,pi)
    """
    Qxx, Qxy = Q[0,0], Q[0,1]
    Qnorm = np.sqrt(Qxx**2 + Qxy**2)
    if Qnorm == 0: return 0, 0
    angle = np.arctan2(Qxy, Qxx)/2
    return Qnorm, angle + np.pi*(angle<0)
    


def draw_nematic_field(nem_field:np.ndarray, box_size:int, **kwargs) -> np.ndarray:
    """Draw nematic field using `nematic_field` function.

    Args:
        nem_field (np.ndarray): nematic field array obtained from `nematic_field`.
        box_size (int): size of boxes
    
    Kwargs:
        uniform_bg (bool): Whether to hide squares and instead opt to colour the lines themselves. Defaults to True.
        lengths (bool): Whether to toggle lengths view instead of colourmap. Defaults to False.

    Returns:
        ndarray: Image of nematic field according to given parameters.
    """
    uniform_bg = True if 'uniform_bg' not in kwargs.keys() else kwargs['uniform_bg']
    lengths = False if 'lengths' not in kwargs.keys() else kwargs['lengths']    
    
    h,w = nem_field.shape[:2] # image shape
    frame = np.zeros((h,w))

    x = y = box_size
    nx, ny = h//x, w//y # number of boxes along height, width
    h_overflow, w_overflow = (h%x), (w%y)
    t_crop, b_crop, l_crop, r_crop = 0,-h,0,-w
    if h_overflow + w_overflow != 0:
        t_crop = h_overflow//2; b_crop = h_overflow-t_crop
        l_crop = w_overflow//2; r_crop = w_overflow-l_crop
        if b_crop == 0: b_crop = -h
        if r_crop == 0: r_crop = -w
    imghelper = np.zeros((h-h_overflow,w-w_overflow)) # of size divisible by x,y,nx,ny ? normally
    h,w = imghelper.shape # cropped shape
    
    if lengths:
        rl, cl = max((min(h/nx,w/ny)//8),1), max((min(h/nx,w/ny)//2),1)
        if rl==1 or cl==1:
            def drawer(t1,t2, Qnorm): # multiply by 1j because rows are imaginary part and columns are real part
                left, right = complex((t1+t2*1j) - 1j*(min(x,y)//4)*np.exp(1j*phi)), complex((t1+t2*1j) + 1j*(min(x,y)//4)*np.exp(1j*phi))
                return sk.draw.line(int(left.real), int(left.imag), int(right.real), int(right.imag))
        else:
            def drawer(t1,t2, Qnorm):
                return sk.draw.ellipse(t1,t2,rl,max(cl*Qnorm,1),(h,w),rotation=phi)   
        # draw lines with different lengths
    else:
        rl, cl = max((min(h/nx,w/ny)//8),1), max((min(h/nx,w/ny)//2),1)
        if rl==1 or cl==1:
            def drawer(t1,t2, Qnorm): # multiply by 1j because rows are imaginary part and columns are real part
                left, right = complex((t1+t2*1j) - 1j*(min(x,y)//4)*np.exp(1j*phi)), complex((t1+t2*1j) + 1j*(min(x,y)//4)*np.exp(1j*phi))
                return sk.draw.line(int(left.real), int(left.imag), int(right.real), int(right.imag))
        else:
            def drawer(t1,t2, Qnorm):
                return sk.draw.ellipse(t1,t2,rl,cl,(h,w),rotation=phi)
    
    boxes = tesselate(nem_field[t_crop:-b_crop,l_crop:-r_crop], box_size, box_size)
        
    maxnorm = np.max([extract(np.mean(box,(0,1)))[0] for box,pos in boxes])/2 if lengths else 1
    # t1 is position in rows, t2 in columbs
    if uniform_bg:
        for box, pos in boxes:
            nem = np.mean(box,(0,1))
            Qnorm, phi = extract(nem)
            t1, t2 = (pos[0]+pos[1])//2, (pos[2]+pos[3])//2
            imghelper[drawer(t1,t2, Qnorm/maxnorm)] = Qnorm
    else:
        for box, pos in boxes:
            nem = np.mean(box,(0,1))
            Qnorm, phi = extract(nem)
            imghelper[pos[0]:pos[1],pos[2]:pos[3]] = Qnorm
            t1, t2 = (pos[0]+pos[1])//2, (pos[2]+pos[3])//2
            imghelper[drawer(t1,t2, Qnorm/maxnorm)] = 0
    frame[t_crop:-b_crop,l_crop:-r_crop] = imghelper
    return frame

def extract_points(nem_field:np.ndarray, mask:np.ndarray, box_size:int, params:list, invert_mask:bool) -> tuple[list, dict]:
    """Generate point data for storing positions and information of boxes.

    Args:
        nem_field (np.ndarray): Nematic field array of shape `(h,w,2,2)`, or `(t,h,w,2,2)`.
        mask (np.ndarray): Mask array of shape `(h,w)`.
        box_size (int): Size of boxes.
        params (list): List of (`str`) parameter names. Must be a subset of `['Qxx', 'Qxy', 'norm', 'angle']`.
        invert_mask (bool): Whether to invert the mask or not. Will change whether mask overlap includes or excludes data.

    Returns:
        tuple[list, dict]: tuple of list and dict corresponding to points positions and properties respectively.
    """
    if 'magicgui.tqdm' in sys.modules:
        from magicgui.tqdm import tqdm
        _tqdm = tqdm
    else:
        from tqdm import tqdm
        _tqdm = tqdm
    def _make_points_for_frame(nem_frame, mask_frame):
        boxes = tesselate(nem_frame[t_crop:-b_crop,l_crop:-r_crop], box_size, box_size)
        points, properties = [], {param:[] for param in params}
        for box, pos in boxes:
            nem = np.mean(box,(0,1))
            Qxx, Qxy = nem[0]
            norm, angle = extract(nem)
            t1, t2 = (pos[0]+pos[1])//2 + t_crop, (pos[2]+pos[3])//2 + l_crop
            if ((mask_frame[t1,t2]>0) and not invert_mask) or (not (mask_frame[t1,t2]>0) and invert_mask):
                points.append([t1,t2])
                if "Qxx" in params: properties["Qxx"].append(Qxx)
                if "Qxy" in params: properties["Qxy"].append(Qxy)
                if "norm" in params: properties["norm"].append(norm)
                if "angle" in params: properties["angle"].append(angle)
        return points, properties

    if len(nem_field.shape) > 4: h,w = nem_field.data.shape[1:3]
    else: h,w = nem_field.data.shape[:2] # image shape    
    
    x = y = box_size
    h_overflow, w_overflow = (h%x), (w%y)
    t_crop, b_crop, l_crop, r_crop = 0,-h,0,-w
    if h_overflow + w_overflow != 0:
        t_crop = h_overflow//2; b_crop = h_overflow-t_crop
        l_crop = w_overflow//2; r_crop = w_overflow-l_crop
        if b_crop == 0: b_crop = -h
        if r_crop == 0: r_crop = -w
    imghelper = np.zeros((h-h_overflow,w-w_overflow)) # of size divisible by x,y,nx,ny ? normally
    h,w = imghelper.shape # cropped shape
    
    
    if len(nem_field.shape) > 4:
        total_points, total_properties = [], {param:[] for param in params}
        for t in _tqdm(range(len(nem_field)), desc='Generating Points...', leave=None):
            points, properties = _make_points_for_frame(nem_field[t], mask[t])
            total_points += [[t]+p for p in points]
            for param_name in params: total_properties[param_name] += properties[param_name]
    else:
        total_points, total_properties = _make_points_for_frame(nem_field, mask)
        
    return total_points, total_properties




if __name__ == '__main__':

    # for __main__ testing purposes
    h,w = 300, 400
    n = 400

    #IMG = make_random_line_image(h,w,n)
    #plt.imshow(IMG)
    #plt.show()
    #sk.io.imsave('test3.png',IMG)
    #A = tesselate(IMG, 3,4)


    IMG = 1*(sk.io.imread('dessin.png') > 0)
    IMG = sk.io.imread('muscle_hard.png')
    IMG = IMG/np.max(IMG)
    if len(IMG.shape) > 2: IMG = sum(IMG.transpose(2,0,1))
    #IMG = pr.thresh(IMG, 0.5)
    
    nem_field = nematic_field(IMG, sigma=0.5, cutoff_ratio=2) # these parameters are very important to control accuracy
    nem_field_image = draw_nematic_field(nem_field, 27)
    

    #fig, (ax1,ax2) = plt.subplots(1,2)
    #ax1.imshow(IMG)
    #ax2.imshow(nem_field)
    #plt.show()

    sk.io.imsave('dessin_p.tif',np.array([IMG, nem_field_image/np.max(nem_field_image)]))
    #sk.io.imsave('dessin_nems.tif',nem_field)
    #sk.io.imsave('muscle_test.tif',np.array([IMG, nem_field_image/np.max(nem_field_image)]))
    #sk.io.imsave('muscle_test_nems.tif',nem_field)
