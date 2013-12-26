# -*- coding: utf-8 -*-
#import models.sxrd_test5_sym_new_test_new66_2_3 as model
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv
import hexahedra,hexahedra_distortion,tetrahedra,octahedra,tetrahedra_edge_distortion,trigonal_pyramid_distortion,trigonal_pyramid_distortion_B,trigonal_pyramid_distortion_B2,trigonal_pyramid_distortion_shareface,trigonal_pyramid_distortion2,trigonal_pyramid_distortion3,trigonal_pyramid_distortion4
import trigonal_pyramid_known_apex

"""
functions in the class
#######################tridentate mode#####################
adding_pb_share_triple: the sorbate is placed at the center point of the plane
adding_pb_share_triple2: the sorbate is plance over the plane, and on the extention line (pass through the center point of plane) from a body center 
adding_pb_share_triple3: the vector from center point to sorbate is normal to the plane
adding_pb_share_triple4: regular trigonal pyramid will be added over a triangle (provide three ref points, will cal a psudo one such that the two and this new one form a equilayer triangle)
adding_share_triple_trigonal_dipyramid: regular hexahedral will be added on top with one middle layer point occupied by lone pair
adding_share_triple_octahedra:regular octahedral (Sb case) will be added ontop (use the same method to cal the third point as adding_Pb_share_triple4)
########################bidentate mode#####################
adding_pb_shareedge: add sorbate (metal, no oxygen) on the extensin line (rooting from body center and through edge center)
adding_sorbate_pyramid_distortion: cal and add sorbates (both metal and oxygen) using function trigonal_pyramid_distortion, edge-distortion is possible
########################monodentate mode###################
adding_sorbate_pyramid_monodentate:metal will be added to right over the attached atm, and the other oxygen atoms will be calculated using function of trigonal_pyramid_known_apex.trigonal_pyramid_two_point
adding_sorbate_bipyramid_monodentate: Pb will be added right over top,the other atoms will be cal by hexahedra.share_corner2()
adding_sorbate_octahedral_monodentate:Sb will be added right over top,the other atoms will be cal by octahedra.share_corner2()
########################outer-sphere#######################
outer_sphere_complex:trigonal_pyramid over crystal surface (could be either apex on top or base on top)
outer_sphere_complex2: the trigonal_pyramid mottif could be rotated by some angle
"""



x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])

#anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
#x2y2z2 are basis of new coor defined in the original frame,new=T.orig
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])

#f2 calculate the distance b/ p1 and p2
f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

#anonymous function f3 is to calculate the coordinates of basis with magnitude of 1.,p1 and p2 are coordinates for two known points, the 
#direction of the basis is pointing from p1 to p2
f3=lambda p1,p2:(1./f2(p1,p2))*(p2-p1)+p1

#extract xyz for atom with id in domain
def extract_coor(domain,id):
    index=np.where(domain.id==id)[0][0]
    x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
    y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
    z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
    return np.array([x,y,z])

def extract_component(domain,id,name_list):
    index=np.where(domain.id==id)[0][0]
    temp=[vars(domain)[name][index] for name in name_list]
    for i in range(len(name_list)):
        print name_list[i]+'=',temp[i]
        
#set coor to atom with id in domain
def set_coor(domain,id,coor):
    index=np.where(domain.id==id)[0][0]
    domain.x[index]=coor[0]
    domain.y[index]=coor[1]
    domain.z[index]=coor[2]

            
#grid matching library for considering offset, x y both from -0.3 to 1.2 with each step of 0.5
#match like 1  2  3
#           6  5  4
#           7  8  9
#the match is based on closest distance
#if you consider match 3 and 6, then 6 will shift towards right by 1 unit to make it to be adjacent to 3, so in this case offset is "+y"
#5 is neighbor to all the other tiles so no offsets (depicted as None)
grid_match_lib={}
grid_match_lib[1]={2:None,3:'-x',4:'-x',5:None,6:None,7:'+y',8:'+y',9:'-x+y'}
grid_match_lib[2]={1:None,3:None,4:None,5:None,6:None,7:'+y',8:'+y',9:'+y'}
grid_match_lib[3]={2:None,1:'+x',4:None,5:None,6:'+x',7:'+x+y',8:'+y',9:'+y'}
grid_match_lib[4]={2:None,3:None,1:'+x',5:None,6:'+x',7:'+x',8:None,9:None}
grid_match_lib[5]={2:None,3:None,4:None,1:None,6:None,7:None,8:None,9:None}
grid_match_lib[6]={2:None,3:'-x',4:'-x',5:None,1:None,7:None,8:None,9:'-x'}
grid_match_lib[7]={2:'-y',3:'-x-y',4:'-x',5:None,6:None,1:'-y',8:None,9:'-x'}
grid_match_lib[8]={2:'-y',3:'-y',4:None,5:None,6:None,7:None,1:'-y',9:None}
grid_match_lib[9]={2:'-y',3:'-y',4:None,5:None,6:'+x',7:'+x',8:None,1:'+x-y'}

class domain_creator_sorbate():
    def __init__(self,ref_domain,id_list,terminated_layer=0,domain_tag='_D1',new_var_module=None):
        #id_list is a list of id in the order of ref_domain,terminated_layer is the index number of layer to be considered
        #for termination,domain_N is a index number for this specific domain, new_var_module is a UserVars module to be used in
        #function of set_new_vars
        self.ref_domain=ref_domain
        self.id_list=id_list
        self.terminated_layer=terminated_layer
        self.domain_tag=domain_tag
        self.share_face,self.share_edge,self.share_corner=(False,False,False)
        #self.anchor_list=[]
        self.polyhedra_list=[]
        self.new_var_module=new_var_module
        self.domain_A,self.domain_B=self.create_equivalent_domains_2()
    
    def build_super_cell(self,ref_domain,rem_atom_ids=None):
    #build a super cell based on the ref_domain, the super cell is actually two domains stacking together in x direction
    #rem_atom_ids is a list of atom ids you want to remove before building a super cell
        super_cell=ref_domain.copy()
        if rem_atom_ids!=None:
            for i in rem_atom_ids:
                super_cell.del_atom(i)
                
        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
            
        for id in super_cell.id:
            index=np.where(ref_domain.id==id)[0][0]
            super_cell.add_atom(id=str(id)+'_+x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
        
        return super_cell
    
    def create_equivalent_domains(self):
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #number 5 here is crystal specific, here is the case for hematite
        for id in self.id_list[:(self.terminated_layer+5)*2]:
            #print id in new_domain_B.id
            new_domain_B.del_atom(id)
        return new_domain_A,new_domain_B

    def create_equivalent_domains_2(self):
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #number 5 here is crystal specific, here is the case for hematite
        for id in self.id_list[:(self.terminated_layer+5)*2]:
            #print id in new_domain_B.id
            new_domain_B.del_atom(id)
        new_domain_A.id=map(lambda x:x+self.domain_tag+'A',new_domain_A.id)
        new_domain_B.id=map(lambda x:x+self.domain_tag+'B',new_domain_B.id)
        return new_domain_A.copy(),new_domain_B.copy()
        
    def adding_pb_share_triple(self,domain,attach_atm_ids=['id1','id2','id3'],offset=[None,None,None],pb_id='pb_id'):
        #the pb will be placed in a plane determined by three points,and lead position is equally distant from the three points
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        p_O3_index=np.where(domain.id==attach_atm_ids[2])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        #try to calculate the center point on the plane, two linear equations based on distance equivalence,one based on point on the plane
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        p_O3=pt_ct(domain,p_O2_index,offset[2])
        p0,p1,p2=p_O1,p_O2,p_O3
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])
        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)
        sorbate_v=center_point
        
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        #return anstrom
        return sorbate_v*basis
        
    def adding_pb_share_triple2(self,domain,r,attach_atm_ids=['id1','id2','id3'],offset=[None,None,None,None],bodycenter_id='Fe',pb_id='pb_id'):
        #the pb will be placed at a point starting from body center of the knonw polyhedra and through a center of a plane determined by three specified points,and lead will be placed somewhere on the extention line
        #r is in angstrom and be counted from the facecenter rather than from the body center
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        p_O3_index=np.where(domain.id==attach_atm_ids[2])
        basis=np.array([5.038,5.434,7.3707])
        body_center_index=np.where(domain.id==bodycenter_id)
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        p_O3=pt_ct(domain,p_O3_index,offset[2])
        p0,p1,p2=p_O1,p_O2,p_O3
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])
        #print p_O1,p_O2,p_O3
        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)
        #sorbate_v=center_point
        
        body_center=pt_ct(domain,body_center_index,offset[3])
        v_bc_fc=(center_point-body_center)*basis
        d_bc_fc=f2(center_point*basis,body_center*basis)
        scalor=(r+d_bc_fc)/d_bc_fc
        sorbate_v=(v_bc_fc*scalor+body_center*basis)/basis
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        #return anstrom
        return sorbate_v*basis
        
    def adding_pb_share_triple3(self,domain,r,attach_atm_ids=['id1','id2','id3'],offset=[None,None,None],pb_id='pb_id'):
        #similar to adding_pb_share_triple2, but no body center, the center point on the plane determined by attach atoms will be the starting point1
        #the pb will be added on the extention line of normal vector (normal to the plane) starting at starting point
        #the distance bt pb and the plane is specified by r, which is in unit of angstrom
        #make sure the order of ids are in anticlock otherwise the sorbate will go inside the bulk structure
        
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        p_O3_index=np.where(domain.id==attach_atm_ids[2])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        #try to calculate the center point on the plane, two linear equations based on distance equivalence,one based on point on the plane
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        p_O3=pt_ct(domain,p_O3_index,offset[2])
        p0,p1,p2=p_O1,p_O2,p_O3
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])

        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)

        normal_scaled=r/(np.dot(normal*basis,normal*basis)**0.5)*normal
        sorbate_v=normal_scaled+center_point
        
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        #return anstrom
        return sorbate_v
     
    def adding_pb_share_triple4(self,domain,top_angle=1.,attach_atm_ids_ref=['id1','id2'],attach_atm_id_third=['id3'],offset=[None,None,None],pb_id='pb_id'):
        #here only consider the angle distortion specified by top_angle (range from 0 to 120 dg), and no length distortion, so the base is a equilayer triangle
        #and here consider the tridentate complexation configuration 
        #two steps:
        #step one: use the coors of the two reference atoms, and the third one to calculate a new third one such that this new point and 
        #the two ref pionts form a equilayer triangle, and the distance from third O to the new third one is shortest
        #see function of _cal_coor_o3 for detail
        #then based on these three points and top angle, calculate the apex coords
        #step two: (has been commented out for this step)
        #update the coord of the third oxygen to the new third coords (be carefule about the offset, you must consider the coor within the unitcell)
        p_O1_index=np.where(domain.id==attach_atm_ids_ref[0])
        p_O2_index=np.where(domain.id==attach_atm_ids_ref[1])
        p_O3_index=np.where(domain.id==attach_atm_id_third[0])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
                            
        def _cal_coor_o3(p0,p1,p3):
            #function to calculate the new point for p3, see document file #2 for detail procedures
            r=f2(p0,p1)/2.*np.tan(np.pi/3)
            norm_vt=p0-p1
            cent_pt=(p0+p1)/2
            a,b,c=norm_vt[0],norm_vt[1],norm_vt[2]
            d=-a*cent_pt[0]-b*cent_pt[1]-c*cent_pt[2]
            u,v,w=p3[0],p3[1],p3[2]
            k=(a*u+b*v+c*w+d)/(a**2+b**2+c**2)
            #projection of O3 to the normal plane see http://www.9math.com/book/projection-point-plane for detail algorithm
            O3_proj=np.array([u-a*k,v-b*k,w-c*k])
            cent_proj_vt=O3_proj-cent_pt
            l=f2(O3_proj,cent_pt)
            ptOnCircle_cent_vt=cent_proj_vt/l*r
            ptOnCircle=ptOnCircle_cent_vt+cent_pt
            return ptOnCircle
 
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        p_O3_old=pt_ct(domain,p_O3_index,offset[2])*basis
        p_O3=_cal_coor_o3(p_O1,p_O2,p_O3_old)
        #print trigonal_pyramid_distortion_shareface
        pyramid_distortion=trigonal_pyramid_distortion_shareface.trigonal_pyramid_distortion_shareface(p0=p_O1,p1=p_O2,p2=p_O3,top_angle=top_angle)
        pyramid_distortion.cal_apex_coor()
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        #dif_value=(p_O3-p_O3_old)/basis
        #domain.dx1[p_O3_index],domain.dy1[p_O3_index],domain.dz1[p_O3_index]=dif_value[0],dif_value[1],dif_value[2]
        #_add_sorbate(domain=domain,id_sorbate=attach_atm_id_third[0],el='O',sorbate_v=(p_O3-_translate_offset_symbols(offset[2]))/basis)
        return pyramid_distortion.apex/basis
    def adding_share_triple_octahedra(self,domain,attach_atm_ids_ref=['id1','id2'],attach_atm_id_third=['id3'],offset=[None,None,None],sorbate_id='Sb_id',sorbate_oxygen_ids=['HO1','HO2','HO3']):
        #here only consider the configuration of regular octahedra
        #and here consider the tridentate complexation configuration 
        #two steps:
        #step one: use the coors of the two reference atoms, and the third one to calculate a new third one such that this new point and 
        #the two ref pionts form a equilayer triangle, and the distance from third O to the new third one is shortest
        #see function of _cal_coor_o3 for detail
        #then based on these three points and top angle, calculate the apex coords
        #step two: 
        #put on the sorbates (center_point +3 oxygens)
        p_O1_index=np.where(domain.id==attach_atm_ids_ref[0])
        p_O2_index=np.where(domain.id==attach_atm_ids_ref[1])
        p_O3_index=np.where(domain.id==attach_atm_id_third[0])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)

                       
        def _cal_coor_o3(p0,p1,p3):
            #function to calculate the new point for p3, see document file #2 for detail procedures
            r=f2(p0,p1)/2.*np.tan(np.pi/3)
            norm_vt=p0-p1
            cent_pt=(p0+p1)/2
            a,b,c=norm_vt[0],norm_vt[1],norm_vt[2]
            d=-a*cent_pt[0]-b*cent_pt[1]-c*cent_pt[2]
            u,v,w=p3[0],p3[1],p3[2]
            k=(a*u+b*v+c*w+d)/(a**2+b**2+c**2)
            #projection of O3 to the normal plane see http://www.9math.com/book/projection-point-plane for detail algorithm
            O3_proj=np.array([u-a*k,v-b*k,w-c*k])
            cent_proj_vt=O3_proj-cent_pt
            l=f2(O3_proj,cent_pt)
            ptOnCircle_cent_vt=cent_proj_vt/l*r
            ptOnCircle=ptOnCircle_cent_vt+cent_pt
            return ptOnCircle
 
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        p_O3_old=pt_ct(domain,p_O3_index,offset[2])*basis
        p_O3=_cal_coor_o3(p_O1,p_O2,p_O3_old)
        #print p_O1,p_O2,p_O3
        octahedra_case=octahedra.share_face(np.array([p_O1,p_O2,p_O3]))
        octahedra_case.share_face_init(flag='regular_triangle')
        #print octahedra_case.center_point
        def _add_sorbate(domain=None,id_sorbate=None,el='Sb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=sorbate_id,el='Sb',sorbate_v=octahedra_case.center_point/basis)
        try:
            _add_sorbate(domain=domain,id_sorbate=sorbate_oxygen_ids[0],el='O',sorbate_v=octahedra_case.p3/basis)
            _add_sorbate(domain=domain,id_sorbate=sorbate_oxygen_ids[1],el='O',sorbate_v=octahedra_case.p4/basis)
            _add_sorbate(domain=domain,id_sorbate=sorbate_oxygen_ids[2],el='O',sorbate_v=octahedra_case.p5/basis)
        except:
            pass
        return [octahedra_case.center_point/basis,octahedra_case.p3/basis,octahedra_case.p4/basis,octahedra_case.p5/basis]
        
    def adding_share_triple_trigonal_bipyramid(self,domain,attach_atm_ids_ref=['id1','id2'],attach_atm_id_third=['id3'],offset=[None,None,None],sorbate_id='Pb_id',sorbate_oxygen_ids=['HO1']):
        #here only consider the configuration of regular hexahedra
        #and here consider the tridentate complexation configuration 
        #two steps:
        #step one: use the coors of the two reference atoms, and the third one to calculate a new third one such that this new point and 
        #the two ref pionts form a equilayer triangle, and the distance from third O to the new third one is shortest
        #see function of _cal_coor_o3 for detail
        #then based on these three points and top angle, calculate the apex coords
        #step two: 
        #put on the sorbates (center_point +1 oxygens with one oxygen left to occupy the lone pair)
        p_O1_index=np.where(domain.id==attach_atm_ids_ref[0])
        p_O2_index=np.where(domain.id==attach_atm_ids_ref[1])
        p_O3_index=np.where(domain.id==attach_atm_id_third[0])
        basis=np.array([5.038,5.434,7.3707])
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index],domain.y[p_O1_index],domain.z[p_O1_index]])+_translate_offset_symbols(symbol)
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)

        #pt_ct2=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0],domain.y[p_O1_index][0],domain.z[p_O1_index][0]])+_translate_offset_symbols(symbol)
                       
        def _cal_coor_o3(p0,p1,p3):
            #function to calculate the new point for p3, see document file #2 for detail procedures
            r=f2(p0,p1)/2.*np.tan(np.pi/3)
            norm_vt=p0-p1
            cent_pt=(p0+p1)/2
            a,b,c=norm_vt[0],norm_vt[1],norm_vt[2]
            d=-a*cent_pt[0]-b*cent_pt[1]-c*cent_pt[2]
            u,v,w=p3[0],p3[1],p3[2]
            k=(a*u+b*v+c*w+d)/(a**2+b**2+c**2)
            #projection of O3 to the normal plane see http://www.9math.com/book/projection-point-plane for detail algorithm
            O3_proj=np.array([u-a*k,v-b*k,w-c*k])
            return O3_proj
 
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        p_O3_old=pt_ct(domain,p_O3_index,offset[2])*basis
        p_O3=_cal_coor_o3(p_O1,p_O2,p_O3_old)
        
        hexahedra_case=hexahedra.share_face(np.array([p_O1,p_O2,p_O3]))
        hexahedra_case.share_face_init(flag='1_2')
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Sb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=sorbate_id,el='Pb',sorbate_v=hexahedra_case.center_point/basis)
        #note the p3 (middle layer type) will be occupied by lone electron pair
        _add_sorbate(domain=domain,id_sorbate=sorbate_oxygen_ids[0],el='O',sorbate_v=hexahedra_case.p4/basis)
        return [hexahedra_case.center_point/basis,hexahedra_case.p4/basis]
        
    def adding_pb_shareedge(self,domain,r=2.,attach_atm_ids=['id1','id2'],offset=[None,None,None],bodycenter_id='Fe',pb_id='pb_id'):
        #the pb will be placed on the extension line from rooting from bodycenter trough edge center
        #note: r is distant in angstrom
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        body_center_index=np.where(domain.id==bodycenter_id)
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        body_center=pt_ct(domain,body_center_index,offset[2])
        
        p1p2_center=(p_O1+p_O2)/2.
        v_bc_ec=(p1p2_center-body_center)*basis
        d_bc_ec=f2(body_center*basis,p1p2_center*basis)
        scalor=(r+d_bc_ec)/d_bc_ec
        sorbate_v=(v_bc_ec*scalor+body_center*basis)/basis
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        return sorbate_v*basis
        

    def adding_sorbate_pyramid_distortion(self,domain,top_angle=1.,phi=0.,edge_offset=[0,0],attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1'],mirror=False,switch=False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        #print "O1",p_O1
        #print "O2",p_O2
        pyramid_distortion=trigonal_pyramid_distortion.trigonal_pyramid_distortion(p0=p_O1,p1=p_O2,top_angle=top_angle,len_offset=edge_offset)
        pyramid_distortion.all_in_all(switch=switch,phi=phi,mirror=mirror)
        #print "apex",pyramid_distortion.apex-[0,0.75587,7.3707]
        #print "p2",pyramid_distortion.p2-[0,0.75587,7.3707]
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        if O_id!=[]:
            _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=pyramid_distortion.p2/basis)
        return [pyramid_distortion.apex/basis,pyramid_distortion.p2/basis]

    def adding_sorbate_pyramid_distortion_B(self,domain,top_angle=1.,phi=0.,edge_offset=[0,0],attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1'],mirror=False,switch=False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        #print "O1",p_O1
        #print "O2",p_O2
        pyramid_distortion=trigonal_pyramid_distortion_B.trigonal_pyramid_distortion(p0=p_O1,p1=p_O2,top_angle=top_angle,len_offset=edge_offset)
        pyramid_distortion.all_in_all(switch=switch,phi=phi,mirror=mirror)
        #print "apex",pyramid_distortion.apex-[0,0.75587,7.3707]
        #print "p2",pyramid_distortion.p2-[0,0.75587,7.3707]
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        if O_id!=[]:
            _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=pyramid_distortion.p2/basis)
        return [pyramid_distortion.apex/basis,pyramid_distortion.p2/basis]
   
    def adding_sorbate_pyramid_distortion_B2(self,domain,top_angle=1.,phi=0.,edge_offset=[0,0],attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1'],mirror=False,switch=False,angle_offset=0):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        #print "O1",p_O1
        #print "O2",p_O2
        pyramid_distortion=trigonal_pyramid_distortion_B2.trigonal_pyramid_distortion(p0=p_O1,p1=p_O2,top_angle=top_angle,len_offset=edge_offset)
        pyramid_distortion.all_in_all(switch=switch,phi=phi,mirror=mirror,angle_offset=angle_offset)
        #print "apex",pyramid_distortion.apex-[0,0.75587,7.3707]
        #print "p2",pyramid_distortion.p2-[0,0.75587,7.3707]
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        if O_id!=[]:
            _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=pyramid_distortion.p2/basis)
        return [pyramid_distortion.apex/basis,pyramid_distortion.p2/basis]
           
    def adding_sorbate_trigonal_bipyramid(self,domain,theta=1.,phi=np.pi/2,flag='1_1+0_1',extend_flag='type1',attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1','id2'],mirror=False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis

        #print "O2",p_O2
        trigonal_bipyramid=hexahedra.share_edge(edge=np.array([p_O1,p_O2]))
        trigonal_bipyramid.all_in_all(theta,phi,None,flag,extend_flag)
        #print "apex",pyramid_distortion.apex-[0,0.75587,7.3707]
        #print "p2",pyramid_distortion.p2-[0,0.75587,7.3707]
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=trigonal_bipyramid.center_point/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=trigonal_bipyramid.p4/basis)
        if mirror:
            _add_sorbate(domain=domain,id_sorbate=O_id[1],el='O',sorbate_v=trigonal_bipyramid.p2/basis)
            return [trigonal_bipyramid.center_point/basis,trigonal_bipyramid.p4/basis,trigonal_bipyramid.p2/basis]
        else:
            _add_sorbate(domain=domain,id_sorbate=O_id[1],el='O',sorbate_v=trigonal_bipyramid.p3/basis)
            return [trigonal_bipyramid.center_point/basis,trigonal_bipyramid.p4/basis,trigonal_bipyramid.p3/basis]
        
    def adding_sorbate_bidentate_octahedral(self,domain,theta=0,phi=np.pi/2,flag='off_center',attach_atm_ids=[],offset=[None,None],sb_id='sb1',O_id=['HO1','HO2','HO3','HO4']):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        #print "O1",p_O1
        #print "O2",p_O2
        octahedral_case=octahedra.share_edge(edge=np.array([p_O1,p_O2]))
        octahedral_case.all_in_all(theta,phi,None,flag)
        #print "apex",pyramid_distortion.apex-[0,0.75587,7.3707]
        #print "p2",pyramid_distortion.p2-[0,0.75587,7.3707]
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=sb_id,el='Sb',sorbate_v=octahedral_case.center_point/basis)
        try:
            _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=octahedral_case.p2/basis)
            _add_sorbate(domain=domain,id_sorbate=O_id[1],el='O',sorbate_v=octahedral_case.p3/basis)
            _add_sorbate(domain=domain,id_sorbate=O_id[2],el='O',sorbate_v=octahedral_case.p4/basis)
            _add_sorbate(domain=domain,id_sorbate=O_id[3],el='O',sorbate_v=octahedral_case.p5/basis)
        except:
            pass
        return [octahedral_case.center_point/basis,octahedral_case.p2/basis,octahedral_case.p3/basis,octahedral_case.p4/basis,octahedral_case.p5/basis]
        
    def adding_sorbate_pyramid_monodentate(self,domain,top_angle=1.,phi=0.,r=2.25,mirror=False,attach_atm_ids=['id1'],offset=None,pb_id='pb_id',O_id=['id1','id2']):
        #The added sorbates (including Pb and one Os) will form a regular trigonal pyramid configuration with the attached ones
        #O-->pb vector is perpendicular to xy plane and the magnitude of this vector is r
        p_O1_index=np.where(domain.id==attach_atm_ids)[0][0]
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #print np.array([domain.x[p_O1_index]+domain.dx1[p_O1_index]+domain.dx2[p_O1_index]+domain.dx3[p_O1_index],domain.y[p_O1_index]+domain.dy1[p_O1_index]+domain.dy2[p_O1_index]+domain.dy3[p_O1_index],domain.z[p_O1_index]+domain.dz1[p_O1_index]+domain.dz2[p_O1_index]+domain.dz3[p_O1_index]]),_translate_offset_symbols(offset)
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index]+domain.dx1[p_O1_index]+domain.dx2[p_O1_index]+domain.dx3[p_O1_index],domain.y[p_O1_index]+domain.dy1[p_O1_index]+domain.dy2[p_O1_index]+domain.dy3[p_O1_index],domain.z[p_O1_index]+domain.dz1[p_O1_index]+domain.dz2[p_O1_index]+domain.dz3[p_O1_index]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset)*basis
        apex=p_O1+[0,0,r]
        pyramid=trigonal_pyramid_known_apex.trigonal_pyramid_two_point(apex=apex,p0=p_O1,top_angle=top_angle,phi=phi,mirror=mirror)
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid.apex/basis)
        if O_id!=[]:
            _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=pyramid.p1/basis)
            _add_sorbate(domain=domain,id_sorbate=O_id[1],el='O',sorbate_v=pyramid.p2/basis)
        return [pyramid.apex/basis,pyramid.p1/basis,pyramid.p2/basis]
        
    def adding_sorbate_bipyramid_monodentate(self,domain,phi=0.,r=2.25,attach_atm_id='id1',offset=None,pb_id='pb_id',O_id=['id1','id2','id3']):
        #The added sorbates (including Pb and one Os) will form a regular hexahedra configuration with the attached ones (also considered the lone pair position)
        #O-->pb vector is perpendicular to xy plane and the magnitude of this vector is r
        p_O1_index=np.where(domain.id==attach_atm_id)
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset)*basis
        bipyramid=hexahedra.share_corner2(corner=p_O1,r=r,phi=phi)
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=bipyramid.center_point/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=bipyramid.p1/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[1],el='O',sorbate_v=bipyramid.p2/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[2],el='O',sorbate_v=bipyramid.p3/basis)
        return [bipyramid.center_point/basis,bipyramid.p1/basis,bipyramid.p2/basis,bipyramid.p3/basis]
        
    def adding_sorbate_octahedral_monodentate(self,domain,phi=0.,r=2.25,attach_atm_id='id1',offset=None,sb_id='sb_id',O_id=['id1','id2','id3','id4','id5']):
        #The added sorbates (including Sb and one Os) will form a regular octahedral configuration with the attached ones 
        #O-->pb vector is perpendicular to xy plane and the magnitude of this vector is r
        p_O1_index=np.where(domain.id==attach_atm_id)[0][0]
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index]+domain.dx1[p_O1_index]+domain.dx2[p_O1_index]+domain.dx3[p_O1_index],domain.y[p_O1_index]+domain.dy1[p_O1_index]+domain.dy2[p_O1_index]+domain.dy3[p_O1_index],domain.z[p_O1_index]+domain.dz1[p_O1_index]+domain.dz2[p_O1_index]+domain.dz3[p_O1_index]])+_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset)*basis
        octahedral_case=octahedra.share_corner2(corner=p_O1,r=r,phi=phi)
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=sb_id,el='Sb',sorbate_v=octahedral_case.center_point/basis)
        try:
            _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=octahedral_case.p1/basis)
            _add_sorbate(domain=domain,id_sorbate=O_id[1],el='O',sorbate_v=octahedral_case.p2/basis)
            _add_sorbate(domain=domain,id_sorbate=O_id[2],el='O',sorbate_v=octahedral_case.p3/basis)
            _add_sorbate(domain=domain,id_sorbate=O_id[3],el='O',sorbate_v=octahedral_case.p4/basis)
            _add_sorbate(domain=domain,id_sorbate=O_id[4],el='O',sorbate_v=octahedral_case.p5/basis) 
        except:
            pass
        return [octahedral_case.center_point/basis,octahedral_case.p1/basis,octahedral_case.p2/basis,octahedral_case.p3/basis,octahedral_case.p4/basis,octahedral_case.p5/basis]
  
    def outer_sphere_complex(self,domain,cent_point=[0.5,0.5,1.],r0=1.,r1=1.,phi=0.,pb_id='pb1',O_ids=['Os1','Os2','Os3']):
        #add a regular trigonal pyramid motiff above the surface representing the outer sphere complexation
        #the pyramid is oriented either Oxygen base top (when r1 is negative) or apex top (when r1 is positive)
        #cent_point in frational coordinate is the center point of the based triangle
        #r0 in ansgtrom is the distance between cent_point and oxygens in the based
        #r1 in angstrom is the distance bw cent_point and apex
        a,b,c=5.038,5.434,7.3707
        p1_x,p1_y,p1_z=r0*np.cos(phi)*np.sin(np.pi/2.)/a+cent_point[0],r0*np.sin(phi)*np.sin(np.pi/2.)/b+cent_point[1],r0*np.cos(np.pi/2.)/c+cent_point[2]
        p2_x,p2_y,p2_z=r0*np.cos(phi+2*np.pi/3)*np.sin(np.pi/2.)/a+cent_point[0],r0*np.sin(phi+2*np.pi/3)*np.sin(np.pi/2.)/b+cent_point[1],r0*np.cos(np.pi/2.)/c+cent_point[2]
        p3_x,p3_y,p3_z=r0*np.cos(phi+4*np.pi/3)*np.sin(np.pi/2.)/a+cent_point[0],r0*np.sin(phi+4*np.pi/3)*np.sin(np.pi/2.)/b+cent_point[1],r0*np.cos(np.pi/2.)/c+cent_point[2]
        apex_x,apex_y,apex_z=cent_point[0],cent_point[1],cent_point[2]+r1/c
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=[apex_x,apex_y,apex_z])
        _add_sorbate(domain=domain,id_sorbate=O_ids[0],el='O',sorbate_v=[p1_x,p1_y,p1_z])
        _add_sorbate(domain=domain,id_sorbate=O_ids[1],el='O',sorbate_v=[p2_x,p2_y,p2_z])
        _add_sorbate(domain=domain,id_sorbate=O_ids[2],el='O',sorbate_v=[p3_x,p3_y,p3_z])
        
    def outer_sphere_complex2(self,domain,cent_point=[0.5,0.5,1.],r0=1.,r1=1.,phi1=0.,phi2=0.,theta=1.57,pb_id='pb1',O_ids=['Os1','Os2','Os3']):
        #different from version 1:consider the orientation of the pyramid, not just up and down
        #add a regular trigonal pyramid motiff above the surface representing the outer sphere complexation
        #the pyramid is oriented either Oxygen base top (when r1 is negative) or apex top (when r1 is positive)
        #cent_point is the fractional coordinates
        #r0 in ansgtrom is the distance between cent_point and oxygens in the based
        #r1 in angstrom is the distance bw cent_point and apex
        
        #anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
        #x2y2z2 are basis of new coor defined in the original frame,new=T.orig
        f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
        #anonymous function f2 to calculate the distance bt two vectors
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        a0_v,b0_v,c0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.]) 
        a,b,c=5.038,5.434,7.3707
        cell=np.array([a,b,c])
        cent_point=cell*cent_point
        p0=np.array(cent_point)
        #first step compute p1, use the original spherical frame origin at center point
        p1_x,p1_y,p1_z=r0*np.cos(phi1)*np.sin(theta)+cent_point[0],r0*np.sin(phi1)*np.sin(theta)+cent_point[1],r0*np.cos(theta)+cent_point[2]
        p1=np.array([p1_x,p1_y,p1_z])
        #step two setup spherical coordinate sys origin at p0
        z_v=(p1-p0)/f2(p0,p1)
        #working on the normal plane, it will crash if z_v[2]==0, check ppt file for detail algorithm
        temp_pt=None
        if z_v[2]!=0:
            temp_pt=np.array([0.,0.,(z_v[1]*p0[1]-z_v[0]*p0[0])/z_v[2]+p0[2]])
        elif z_v[1]!=0:
            temp_pt=np.array([0.,(z_v[2]*p0[2]-z_v[0]*p0[0])/z_v[1]+p0[1],0.])
        else:
            temp_pt=np.array([(-z_v[2]*p0[2]-z_v[1]*p0[1])/z_v[0]+p0[0],0.,0.])
        x_v=(temp_pt-p0)/f2(temp_pt,p0)
        y_v=np.cross(z_v,x_v)
        T=f1(a0_v,b0_v,c0_v,x_v,y_v,z_v)
        #then calculte p2, note using the fact p2p0 is 120 degree apart from p1p0, since the base is equilayer triangle
        p2_x,p2_y,p2_z=r0*np.cos(phi2)*np.sin(np.pi*2./3.),r0*np.sin(phi2)*np.sin(np.pi*2./3.),r0*np.cos(np.pi*2./3.)
        p2_new=np.array([p2_x,p2_y,p2_z])
        p2=np.dot(inv(T),p2_new)+p0
        #step three calculate p3, use the fact p3 on the vector extension of p1p2cent_p0
        p3=(p0-(p1+p2)/2.)*3+(p1+p2)/2.
        #step four calculate p4, cross product, note the magnitute here is in angstrom, so be careful
        p4_=np.cross(p2-p0,p1-p0)
        zero_v=np.array([0,0,0])
        p4=p4_/f2(p4_,zero_v)*r1+p0
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=p4/cell)
        _add_sorbate(domain=domain,id_sorbate=O_ids[0],el='O',sorbate_v=p1/cell)
        _add_sorbate(domain=domain,id_sorbate=O_ids[1],el='O',sorbate_v=p2/cell)
        _add_sorbate(domain=domain,id_sorbate=O_ids[2],el='O',sorbate_v=p3/cell)      
    
 