import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline.

    Parameters:
    -----------
    Kl   - 3 x 3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3 x 3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---
    
    # Compute baseline (right camera translation minus left camera translation).
    b = Twl[0:3,3] - Twr[0:3,3]
    
    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
    rayl = Twl[0:3,0:3].dot ( inv(Kl).dot( np.vstack([pl,1]) ) ) 
    norm_rl = norm(rayl)
    rayl = rayl / norm_rl
    
    rayr = Twr[0:3,0:3].dot ( inv(Kr).dot( np.vstack([pr,1]) ) ) 
    norm_rr = norm(rayr)
    rayr = rayr / norm_rr
    
    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    ml = ( b.dot(rayl) -  (b.dot(rayr) * rayl.T.dot(rayr) )  ) / (1 -  (rayl.T.dot(rayr))**2 )
    mr = rayl.T.dot(rayr) * ml - b.dot(rayr)
    
    
    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.
    Pl = Twl[0:3,3:4] + rayl * ml
    Pr = Twr[0:3,3:4] + rayr * mr
    
    
    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.
    
    # Add code here...
    
    #calculate the derivative of r_hat
    
    #derivative of the scaling factor for u and v
    dulscalar = (1/2) * ( ( norm(pl) ) ** (-1/2) ) * 2 * ( pl[0,0] )
    dvlscalar = (1/2) * ( ( norm(pl) ) ** (-1/2) ) * 2 * ( pl[1,0] )
    #derivative of RL w.r.t u and v, using quotient rule 
    dul = (Twl[0:3,0:3].dot( inv(Kl)[0:3,0:1] ) * norm_rl -  rayl * dulscalar) / (norm_rl ** 2)
    dvl = (Twl[0:3,0:3].dot( inv(Kl)[0:3,1:2] ) * norm_rl -  rayl * dvlscalar) / (norm_rl ** 2)
    #derivate of RL w.r.t ur and vr are 0
    dur = np.zeros(shape = (3,1))
    dvr = np.zeros(shape = (3,1))
    #putting them together
    drayl = np.hstack([dul,dvl,dur,dvr])
    
    #derivative of the scaling factor for u and v
    durscalar = (1/2) * ( ( norm(pr) ) ** (-1/2) ) * 2 * ( pr[0,0] )
    dvrscalar = (1/2) * ( ( norm(pr) ) ** (-1/2) ) * 2 * ( pr[1,0] )
    #derivative of RR w.r.t u and v, using quotient rule
    dur = (Twr[0:3,0:3].dot( inv(Kr)[0:3,0:1] ) * norm_rr -  rayr * durscalar) / (norm_rr ** 2)
    dvr = (Twr[0:3,0:3].dot( inv(Kr)[0:3,1:2] ) * norm_rr -  rayr * dvrscalar) / (norm_rr ** 2)
    #derivate of RR w.r.t ul and vl are 0
    dul = np.zeros(shape = (3,1))
    dvl = np.zeros(shape = (3,1))
    #putting them together
    drayr = np.hstack([dul,dvl,dur,dvr])
    #------------------
    
    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2
    
    du = (b.T@drayl).reshape(1, 4) - \
            (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
            np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))
    
    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1
    
    dm = (b.T@drayr).reshape(1, 4) - \
            (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
            np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv
    
    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2
    
    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2
    
    #--- FILL ME IN ---
    # 3D point.
    P = (Pl + Pr) / 2
    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    S = np.hstack([Sl, np.zeros(shape = (2,2)) ])
    S = np.vstack( [S,np.hstack([np.zeros(shape = (2,2)), Sr ])] )
    
    #------------------

    return Pl, Pr, P, S
