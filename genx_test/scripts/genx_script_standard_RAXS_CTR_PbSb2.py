import models.sxrd_new1 as model
import models.raxs as model2
from models.utils import UserVars
from datetime import datetime
import numpy as np
import sys,pickle,__main__
import models.domain_creator as domain_creator
sys.path.append("D:\Google Drive\useful codes")
try:
    import make_parameter_table_GenX as make_grid
except:pass
from copy import deepcopy

#************************************program begins from here **********************************************
###############################################global vars##################################################
COUNT_TIME=False

if COUNT_TIME:t_0=datetime.now()

##file paths and wt factors##
batch_path_head='/u1/uaf/cqiu/batchfile/'
WT_RAXS=5#weighting for RAXS dataset
WT_BV=1#weighting for bond valence constrain (1 recommended)
BV_TOLERANCE=0.05#ideal bv value + or - this value is acceptable

##pars for sorbates##
SORBATE=["Pb","Sb"]#any combo of "Pb" and "Sb"
SORBATE_NUMBER=[[1,0],[0,1]]
O_NUMBER=[[[1],[0]],[[0],[0]]]#either zero oxygen ligand or enough ligands to complete coordinative shell
SORBATE_LIST=domain_creator.create_sorbate_el_list(SORBATE,SORBATE_NUMBER)
BV_SUM=[[1.33],[5.]]#pseudo bond valence sum for sorbate
SORBATE_ATTACH_ATOM=[[['O1_1_0','O1_2_0']],[['O1_1_0','O1_2_0','O1_3_0']]]
SORBATE_ATTACH_ATOM_OFFSET=[[[None,None]],[[None,None,None]]]
ANCHOR_REFERENCE=[[None],[None]]#ref point for anchors
ANCHOR_REFERENCE_OFFSET=[[None],[None]]
DISCONNECT_BV_CONTRIBUTION=[{('O1_1_0','O1_2_0'):'Pb1'},{}]#set items to be {} if considering single sorbate
#if consider hydrogen bonds#
COVALENT_HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0'],['O1_1_0','O1_2_0']]
COVALENT_HYDROGEN_NUMBER=[[1,1],[1,1]]
POTENTIAL_HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0'],['O1_1_0','O1_2_0']]#they can accept one hydrogen bond or not
#geometrical parameters#
TOP_ANGLE=[[1.38],[1.38]]
PHI=[[0],[0]]
R_S=[[1],[1]]
MIRROR=False

##pars for interfacial waters##
WATER_NUMBER=[0,0]
REF_POINTS=[['HO1','HO2'],['O1_1_0','O1_2_0']]#each item inside is a list of one or couple items
V_SHIFT=[[2.8],[2]]
R=[[1.5],[3]]
ALPHA=[[0],[3]]

##chemically different domain type##
DOMAIN=[1,1]
DOMAIN_GP=[[0,1]]#means you want to group first two and last two domains together, only group half layers or full layers together
DOMAIN_NUMBER=len(DOMAIN)
COHERENCE=[True,True] #want to add up in coherence?

##cal bond valence switch##
USE_BV=True
DOMAINS_BV=[0,1]#Domains being considered for bond valence constrain, counted from 0

##want to output the data for plotting?##
PLOT=False

##want to make parameter table?##
TABLE=False
if TABLE:
    O_N=[]
    for i in O_NUMBER:
        temp=0
        for j in i:
            temp+=sum(j)
        O_N.append([temp])
    make_grid.make_structure(map(sum,SORBATE_NUMBER),O_N,TOP_ANGLE,PHI,WATER_NUMBER,DOMAIN,Metal=SORBATE[1])

##############################################set up atm ids###############################################

for i in range(DOMAIN_NUMBER):
    ##text files
    vars()['discrete_vars_file_domain'+str(int(i+1))]='new_varial_file_domain'+str(int(i+1))+'.txt'
    
    ##user defined variables
    vars()['rgh_domain'+str(int(i+1))]=UserVars()
    
    ##sorbate list (HO is oxygen binded to pb and Os is water molecule)
    vars()['SORBATE_list_domain'+str(int(i+1))+'a']=domain_creator.create_sorbate_ids2(el=SORBATE,N=SORBATE_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['SORBATE_list_domain'+str(int(i+1))+'b']=domain_creator.create_sorbate_ids2(el=SORBATE,N=SORBATE_NUMBER[i],tag='_D'+str(int(i+1))+'B')
    #print SORBATE_list_domain1b
    vars()['HO_list_domain'+str(int(i+1))+'a']=domain_creator.create_HO_ids2(anchor_els=SORBATE,O_N=O_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['HO_list_domain'+str(int(i+1))+'b']=domain_creator.create_HO_ids2(anchor_els=SORBATE,O_N=O_NUMBER[i],tag='_D'+str(int(i+1))+'B')
    
    vars()['Os_list_domain'+str(int(i+1))+'a']=domain_creator.create_sorbate_ids(el='Os',N=WATER_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['Os_list_domain'+str(int(i+1))+'b']=domain_creator.create_sorbate_ids(el='Os',N=WATER_NUMBER[i],tag='_D'+str(int(i+1))+'B')     
    
    vars()['sorbate_ids_domain'+str(int(i+1))+'a']=vars()['SORBATE_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']+vars()['Os_list_domain'+str(int(i+1))+'a']
    vars()['sorbate_ids_domain'+str(int(i+1))+'b']=vars()['SORBATE_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']+vars()['Os_list_domain'+str(int(i+1))+'b']
    vars()['sorbate_els_domain'+str(int(i+1))]=SORBATE_LIST[i]+['O']*(sum([np.sum(N_list) for N_list in O_NUMBER[i]])+WATER_NUMBER[i])

    ##set up group name container(discrete:single atom from each domain, sequence:double atoms at same layer from each domain)
    #atom ids for grouping(containerB must be the associated chemically equivalent atoms)
    equivalent_atm_list_A_1=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0"]
    equivalent_atm_list_A_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1"]
    vars()['ids_domain'+str(int(i+1))+'A']=vars()['sorbate_ids_domain'+str(int(i+1))+'a']+map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['equivalent_atm_list_A_'+str(int(DOMAIN[i]))])
    equivalent_atm_list_B_1=["O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1"]
    equivalent_atm_list_B_2=["O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1","O1_7_1","O1_8_1","Fe1_8_1","Fe1_9_1","O1_9_1","O1_10_1","Fe1_10_1","Fe1_12_1"]
    vars()['ids_domain'+str(int(i+1))+'B']=vars()['sorbate_ids_domain'+str(int(i+1))+'b']+map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['equivalent_atm_list_B_'+str(int(DOMAIN[i]))])

    vars()['discrete_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x.rsplit('_')[0]+'_D'+str(int(i+1)),vars()['sorbate_ids_domain'+str(int(i+1))+'a'])+\
                                                     map(lambda x:'gp_'+x[0].rsplit('_')[0][:-1]+x[0].rsplit('_')[1]+x[1].rsplit('_')[0][:-1]+x[1].rsplit('_')[1]+'_D'+str(int(i+1)),zip(vars()['equivalent_atm_list_A_'+str(int(DOMAIN[i]))],vars()['equivalent_atm_list_B_'+str(int(DOMAIN[i]))]))
    #consider the top 10 atom layers
    atm_sequence_gp_names_1=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6']
    atm_sequence_gp_names_2=['O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12']
    vars()['sequence_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x+'_D'+str(int(i+1)),vars()['atm_sequence_gp_names_'+str(int(DOMAIN[i]))])
    
    ##atom ids being considered for bond valence check
    atm_list_A_1=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','Fe1_4_0','Fe1_6_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0']
    atm_list_A_2=['O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0','Fe1_4_1','Fe1_6_1']
    
    atm_list_B_1=['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_10_0','Fe1_12_0','Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1']
    atm_list_B_2=["O1_11_0","O1_12_0","O1_1_1","O1_2_1",'O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1','Fe1_8_1','Fe1_9_1']
    
    vars()['atm_list_'+str(int(i+1))+'A']=map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['atm_list_A_'+str(int(DOMAIN[i]))])
    vars()['atm_list_'+str(int(i+1))+'B']=map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['atm_list_B_'+str(int(DOMAIN[i]))])

##id list according to the order in the reference domain (used to set up ref domain)  
ref_id_list_1=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
ref_id_list_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']

###############################################setting slabs##################################################################    
unitcell = model.UnitCell(5.038, 5.434, 7.3707, 90, 90, 90)
inst = model.Instrument(wavel = .833, alpha = 2.0)
bulk = model.Slab(T_factor='B')
ref_domain1 =  model.Slab(c = 1.0,T_factor='B')
ref_domain2 =  model.Slab(c = 1.0,T_factor='B')
rgh=UserVars()
rgh.new_var('beta', 0.0)
scales=['scale_CTR','scale_RAXS','scale_CTR_specular']
for scale in scales:
    rgh.new_var(scale,1.)
    
################################################build up ref domains############################################
#add atoms for bulk and two ref domains (ref_domain1<half layer> and ref_domain2<full layer>)
#In those two reference domains, the atoms are ordered according to first hight (z values), then y values
#it is a super surface structure by stacking the surface slab on bulk slab, the repeat vector was counted 

#only two possible path(one for runing in pacman, the other in local laptop)
try:
    domain_creator.add_atom_in_slab(bulk,batch_path_head+'bulk.str')
except:
    batch_path_head='\\'.join(__main__.__file__.rsplit('\\')[:-1])+'\\batchfile\\'
    domain_creator.add_atom_in_slab(bulk,batch_path_head+'bulk.str')
domain_creator.add_atom_in_slab(ref_domain1,batch_path_head+'half_layer2.str')
domain_creator.add_atom_in_slab(ref_domain2,batch_path_head+'full_layer3.str')

###################create domain classes and initiate the chemical equivalent domains####################
#when change or create a new domain, make sure the terminated_layer (start from 0)set right

##setup domains
for i in range(DOMAIN_NUMBER):
    vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
    vars()['domain'+str(int(i+1))+'A']=vars()['domain_class_'+str(int(i+1))].domain_A
    vars()['domain'+str(int(i+1))+'B']=vars()['domain_class_'+str(int(i+1))].domain_B
    vars(vars()['domain_class_'+str(int(i+1))])['domainA']=vars()['domain'+str(int(i+1))+'A']
    vars(vars()['domain_class_'+str(int(i+1))])['domainB']=vars()['domain'+str(int(i+1))+'B']
    #Adding sorbates to domainA and domainB
    for j in range(sum(SORBATE_NUMBER[i])):
        SORBATE_coors_a=[]
        O_coors_a=[]
        #jj=SORBATE_LIST[i].index(SORBATE_LIST[i][j])
        #here consider a case like SORBATE_LIST=['Pb','Pb']
        #if len(filter(lambda x:x==SORBATE_LIST[i][j],SORBATE_LIST[i]))>1:
        #    jj=j
        if len(SORBATE_ATTACH_ATOM[i][j])==1:#monodentate case
            ids=SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A'
            #print ids
            offset=SORBATE_ATTACH_ATOM_OFFSET[i][j][0]
            SORBATE_id=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]#pb_id is a str NOT list
            #O_index=[0]+[sum(O_NUMBER[i][0:ii+1]) for ii in range(len(O_NUMBER[i]))]
            #for [1,2,2], which means inside one domain there are 1OH corresponding to pb1, 2 OH's corresponding to pb2 and so son.
            #will return [0,1,3,5], O_id extract OH according to O_index
            O_id=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
            #print SORBATE_id,O_id,vars()['HO_list_domain'+str(int(i+1))+'a']
            #O_id=vars()['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]#O_ide is a list of str
            sorbate_coors=[]
            if SORBATE_LIST[i][j]=='Pb':
                sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_monodentate(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=TOP_ANGLE[i][j],phi=PHI[i][j],r=R_S[i][j],attach_atm_ids=ids,offset=offset,pb_id=SORBATE_id,O_id=O_id,mirror=MIRROR)           
            elif SORBATE_LIST[i][j]=='Sb':
                sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_octahedral_monodentate(domain=vars()['domain'+str(int(i+1))+'A'],phi=PHI[i][j],r=R_S[i][j],attach_atm_id=ids,offset=offset,sb_id=SORBATE_id,O_id=O_id)           
            SORBATE_coors_a.append(sorbate_coors[0])
            if O_id!=[]:
                [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
            SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
            O_id_B=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
            #now put on sorbate on the symmetrically related domain
            sorbate_ids=[SORBATE_id_B]+O_id_B
            sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
            domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
            #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
            sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
            HO_set_ids=O_id+O_id_B
            N=len(sorbate_set_ids)/2
            M=len(O_id)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
            #if O_NUMBER[i][j]!=0:
            if M!=0:
                vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=HO_set_ids)
        elif len(SORBATE_ATTACH_ATOM[i][j])==2:#bidentate case
            ids=[SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A']
            offset=SORBATE_ATTACH_ATOM_OFFSET[i][j]
            anchor,anchor_offset=None,None
            if ANCHOR_REFERENCE[i][j]!=None:
                anchor=ANCHOR_REFERENCE[i][j]+'_D'+str(int(i+1))+'A'
                anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]
            SORBATE_id=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
            #O_index=[0]+[sum(O_NUMBER[i][0:ii+1]) for ii in range(len(O_NUMBER[i]))]
            #if O_index[0:2]==[0,0]:
            #    O_index=O_index[1:]
            #O_id=vars()['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]
            O_id=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
            sorbate_coors=[]
            if SORBATE_LIST[i][j]=='Pb':
                sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_distortion_B(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=TOP_ANGLE[i][j],phi=PHI[i][j],edge_offset=[0,0],attach_atm_ids=ids,offset=offset,anchor_ref=anchor,anchor_offset=anchor_offset,pb_id=SORBATE_id,O_id=O_id,mirror=MIRROR)
            elif SORBATE_LIST[i][j]=='Sb':
                sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_octahedral(domain=vars()['domain'+str(int(i+1))+'A'],theta=PHI[i][j],phi=TOP_ANGLE[i][j],attach_atm_ids=ids,offset=offset,sb_id=SORBATE_id,O_id=O_id)
            SORBATE_coors_a.append(sorbate_coors[0])
            #if O_NUMBER[i][j]!=0:
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
            SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
            O_id_B=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
            #now put on sorbate on the symmetrically related domain
            sorbate_ids=[SORBATE_id_B]+O_id_B
            sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
            domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
            #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
            sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
            HO_set_ids=O_id+O_id_B
            N=len(sorbate_set_ids)/2
            M=len(O_id)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
            if M!=0:
                vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=HO_set_ids)
        elif len(SORBATE_ATTACH_ATOM[i][j])==3:#tridentate case (no oxygen sorbate here considering it is a trigonal pyramid structure)
            ids=[SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][2]+'_D'+str(int(i+1))+'A']
            offset=SORBATE_ATTACH_ATOM_OFFSET[i][j]
            SORBATE_id=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
            O_index,O_id,sorbate_coors,O_id_B,HO_set_ids,SORBATE_id_B,sorbate_ids,SORBATE_coors_a=[],[],[],[],[],[],[],[]
            if SORBATE_LIST[i][j]=='Sb':
                #O_index=[0]+[sum(O_NUMBER[i][0:ii+1]) for ii in range(len(O_NUMBER[i]))]
                #O_id=vars()['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]
                O_id=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_share_triple_octahedra(domain=vars()['domain'+str(int(i+1))+'A'],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_oxygen_ids=O_id)
                SORBATE_coors_a.append(sorbate_coors[0])
                [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                O_id_B=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                sorbate_ids=[SORBATE_id_B]+O_id_B
                sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)

            elif SORBATE_LIST[i][j]=='Pb':
                sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_pb_share_triple4(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=TOP_ANGLE[i][j],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,pb_id=SORBATE_id)
                SORBATE_coors_a.append(sorbate_coors)
                SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                #now put on sorbate on the symmetrically related domain
                sorbate_ids=[SORBATE_id_B]
                sorbate_els=[SORBATE_LIST[i][j]]
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
            #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
            if SORBATE_LIST[i][j]=='Sb':
                sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
                HO_set_ids=O_id+O_id_B
                N=len(sorbate_set_ids)/2
                M=len(O_id)
                vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
                if M!=0:
                    vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=HO_set_ids)
            elif SORBATE_LIST[i][j]=='Pb':
                sorbate_set_ids=[SORBATE_id]+[SORBATE_id_B]
                N=len(sorbate_set_ids)/2
                vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)

    if WATER_NUMBER[i]!=0:#add water molecules if any
        for jj in range(WATER_NUMBER[i]/2):#note will add water pair (two oxygens) each time, and you can't add single water 
            O_ids_a=vars()['Os_list_domain'+str(int(i+1))+'a'][jj*2:jj*2+2]
            O_ids_b=vars()['Os_list_domain'+str(int(i+1))+'b'][jj*2:jj*2+2]
            #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
            H2O_coors_a=vars()['domain_class_'+str(int(i+1))].add_oxygen_pair2(domain=vars()['domain'+str(int(i+1))+'A'],O_ids=O_ids_a,ref_id=map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i]),v_shift=V_SHIFT[i][jj],r=R[i][jj],alpha=ALPHA[i][jj])
            add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O','O'])
            #group water molecules at each layer (set equivalent the oc and u during fitting)
            M=len(O_ids_a)
            vars()['gp_waters_set'+str(jj+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=O_ids_a+O_ids_b)
    #set variables
    vars()['domain_class_'+str(int(i+1))].set_discrete_new_vars_batch(batch_path_head+vars()['discrete_vars_file_domain'+str(int(i+1))])
    
######################################do grouping###############################################
for i in range(DOMAIN_NUMBER):
    #note the grouping here is on a layer basis, ie atoms of same layer are groupped together (4 atms grouped together in sequence grouping)
    #you may group in symmetry, then atoms of same layer are not independent (and know the use_sym set to false here)
    if DOMAIN[i]==1:
        vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_1_0_D'+str(int(i+1))+'A','O1_7_0_D'+str(int(i+1))+'B']],layers_N=10)
    elif DOMAIN[i]==2:
        vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_5_0_D'+str(int(i+1))+'A','O1_11_0_D'+str(int(i+1))+'B']],layers_N=10)
    
    #assign name to each group
    for j in range(len(vars()['sequence_gp_names_domain'+str(int(i+1))])):vars()[vars()['sequence_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_list_domain'+str(int(i+1))][j]
    
    #you may also only want to group each chemically equivalent atom from two domains (the use_sym is set to true here)
    vars()['atm_gp_discrete_list_domain'+str(int(i+1))]=[]
    for j in range(len(vars()['ids_domain'+str(int(i+1))+'A'])):
        vars()['atm_gp_discrete_list_domain'+str(int(i+1))].append(vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']],\
                                                                   atom_ids=[vars()['ids_domain'+str(int(i+1))+'A'][j],vars()['ids_domain'+str(int(i+1))+'B'][j]],sym_array=[[1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.]]))
    for j in range(len(vars()['discrete_gp_names_domain'+str(int(i+1))])):vars()[vars()['discrete_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))][j]

#gp_Pb_D1D2=domain_class_1.grouping_discrete_layer3(domain=[domain1A,domain1B,domain2A,domain2B],atom_ids=['Pb1_D1A','Pb1_D1B','Pb1_D2A','Pb1_D2B'])
#gp_Pb_D1=domain_class_1.grouping_discrete_layer([domain1A,domain1B,domain1A,domain1B],['Pb1_D1A','Pb1_D1B','Pb2_D1A','Pb2_D1B'])

for group in DOMAIN_GP:
    a,b=group[0]+1,group[1]+1
    if DOMAIN[group[0]]==1:
        vars()['sequence_gp_list_domain_'+str(a)+'_'+str(b)]=domain_class_1.grouping_sequence_layer_new(domain=[[vars()['domain'+str(a)+'A'],vars()['domain'+str(a)+'B'],vars()['domain'+str(b)+'A'],vars()['domain'+str(b)+'B']]], first_atom_id=[['O1_1_0_D'+str(a)+'A','O1_7_0_D'+str(a)+'B','O1_1_0_D'+str(b)+'A','O1_7_0_D'+str(b)+'B']],layers_N=10)
    elif DOMAIN[group[0]]==2:
        vars()['sequence_gp_list_domain_'+str(a)+'_'+str(b)]=domain_class_1.grouping_sequence_layer_new(domain=[[vars()['domain'+str(a)+'A'],vars()['domain'+str(a)+'B'],vars()['domain'+str(b)+'A'],vars()['domain'+str(b)+'B']]], first_atom_id=[['O1_5_0_D'+str(a)+'A','O1_11_0_D'+str(a)+'B','O1_5_0_D'+str(b)+'A','O1_11_0_D'+str(b)+'B']],layers_N=10)
    vars()['discrete_gp_list_domain_'+str(a)+'_'+str(b)]=[]
    def _match(value):
        if value==0:return 1
        else:return -1
    for j in range(len(vars()['atm_list_'+str(a)+'A'])):
        vars()['discrete_gp_list_domain_'+str(a)+'_'+str(b)].append(domain_class_1.grouping_discrete_layer2(domain=[vars()['domain'+str(a)+'A'],vars()['domain'+str(a)+'B'],vars()['domain'+str(b)+'A'],vars()['domain'+str(b)+'B']],\
                                                                   atom_ids=[vars()['atm_list_'+str(a)+'A'][j],vars()['atm_list_'+str(a)+'B'][j],vars()['atm_list_'+str(b)+'A'][j+_match(j%2)],vars()['atm_list_'+str(b)+'B'][j+_match(j%2)]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1],[1,0,0,0,1,0,0,0,1]],use_sym=True))
    #assign names to each super atm group 
    def _reorder(original_list):
        new_list=[]
        N=len(original_list)
        index_list_1=[ii*2+1 for ii in range(N/2)]
        index_list_2=[ii*2 for ii in range(N/2)]
        for i in range(N/2):
            new_list.append(original_list[index_list_1[i]])
            new_list.append(original_list[index_list_2[i]])
        return new_list
        
    vars()['discrete_gp_names_domain_'+str(a)+'_'+str(b)]=map(lambda x:'gp_'+x[0].rsplit('_')[0][:-1]+x[0].rsplit('_')[1]+x[1].rsplit('_')[0][:-1]+x[1].rsplit('_')[1]+'_D'+str(a)+'_'+x[2].rsplit('_')[0][:-1]+x[2].rsplit('_')[1]+x[3].rsplit('_')[0][:-1]+x[3].rsplit('_')[1]+'_D'+str(b),zip(vars()['equivalent_atm_list_A_'+str(DOMAIN[group[0]])],vars()['equivalent_atm_list_B_'+str(DOMAIN[group[0]])],_reorder(vars()['equivalent_atm_list_A_'+str(DOMAIN[group[0]])]),_reorder(vars()['equivalent_atm_list_B_'+str(DOMAIN[group[0]])])))
    vars()['sequence_gp_names_domain_'+str(a)+'_'+str(b)]=['gp_sequence_layer'+str(x+1)+'_'+'D'+str(a)+'_D'+str(b) for x in range(10)]
    for j in range(len(vars()['sequence_gp_list_domain_'+str(a)+'_'+str(b)])):vars()[vars()['sequence_gp_names_domain_'+str(a)+'_'+str(b)][j]]=vars()['sequence_gp_list_domain_'+str(a)+'_'+str(b)][j]
    for j in range(len(vars()['discrete_gp_list_domain_'+str(a)+'_'+str(b)])):vars()[vars()['discrete_gp_names_domain_'+str(a)+'_'+str(b)][j]]=vars()['discrete_gp_list_domain_'+str(a)+'_'+str(b)][j]

#####################################do bond valence matching###################################
if USE_BV:
    for i in range(DOMAIN_NUMBER):
        lib_sorbate={}
        if SORBATE_NUMBER[i]!=0:
            lib_sorbate=domain_creator.create_sorbate_match_lib4(metal=SORBATE_LIST[i],HO_list=vars()['HO_list_domain'+str(int(i+1))+'a'],anchors=SORBATE_ATTACH_ATOM[i],anchor_offsets=SORBATE_ATTACH_ATOM_OFFSET[i],domain_tag=i+1)
        if DOMAIN[i]==1:
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=['Fe1_2_0_D'+str(int(i+1))+'A','Fe1_3_0_D'+str(int(i+1))+'A']),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.merge_two_libs(vars()['match_lib_'+str(int(i+1))+'A'],lib_sorbate)
        elif DOMAIN[i]==2:
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=None),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.merge_two_libs(vars()['match_lib_'+str(int(i+1))+'A'],lib_sorbate)
        if DISCONNECT_BV_CONTRIBUTION[i]!={}:
            ids=map(lambda x:x+'_D'+str(i+1)+'A',DISCONNECT_BV_CONTRIBUTION[i].keys()[0])
            value=DISCONNECT_BV_CONTRIBUTION[i][DISCONNECT_BV_CONTRIBUTION[i].keys()[0]]
            for id in ids:
                index_temp=vars()['match_lib_'+str(int(i+1))+'A'][id].index(filter(lambda x:value in x,vars()['match_lib_'+str(int(i+1))+'A'][id])[0])
                del vars()['match_lib_'+str(int(i+1))+'A'][id][index_temp]
#####################################specify f1f2 here###################################
res_el='Pb'
f1f2_file='raxs_Pb_formatted.f1f2'
#f1f2=np.loadtxt(batch_path_head+f1f2_file)
VARS=vars()#pass local variables to sim function
###################################fitting function part##########################################
if COUNT_TIME:t_1=datetime.now()

def Sim(data,VARS=VARS):
    VARS=VARS
    F =[]
    bv=0
    bv_container={}
    debug_bv=False
    fom_scaler=[]
    beta=rgh.beta
    SCALES=[getattr(rgh,scale) for scale in scales]
    total_wt=0
    domain={}

    for i in range(DOMAIN_NUMBER):
        #extract the fitting par values in the associated attribute and then do the scaling(initiation+processing, actually update the fitting parameter values)
        #VARS['domain_class_'+str(int(i+1))].init_sim_batch(batch_path_head+VARS['sim_batch_file_domain'+str(int(i+1))])
        #VARS['domain_class_'+str(int(i+1))].scale_opt_batch(batch_path_head+VARS['scale_operation_file_domain'+str(int(i+1))])   
        #grap wt for each domain and cal the total wt
        vars()['wt_domain'+str(int(i+1))]=VARS['rgh_domain'+str(int(i+1))].wt
        total_wt=total_wt+vars()['wt_domain'+str(int(i+1))]

        #update sorbates
        for j in range(sum(VARS['SORBATE_NUMBER'][i])):
            SORBATE_coors_a=[]
            O_coors_a=[]
            #jj=SORBATE_LIST[i].index(SORBATE_LIST[i][j])
            #here consider a case like SORBATE_LIST=['Pb','Pb']
            #if len(filter(lambda x:x==SORBATE_LIST[i][j],SORBATE_LIST[i]))>1:
            #    jj=j
            if len(VARS['SORBATE_ATTACH_ATOM'][i][j])==1:#monodentate case
                top_angle=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle')
                phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi')
                r=getattr(VARS['rgh_domain'+str(int(i+1))],'r')
                ids=VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A'
                offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j][0]
                SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]#pb_id is a str NOT list
                #O_index=[0]+[sum(VARS['O_NUMBER'][i][0:ii+1]) for ii in range(len(VARS['O_NUMBER'][i]))]
                #for [1,2,2], which means inside one domain there are 1OH corresponding to pb1, 2 OH's corresponding to pb2 and so son.
                #will return [0,1,3,5], O_id extract OH according to O_index
                O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                #O_id=VARS['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]#O_ide is a list of str
                sorbate_coors=[]
                if SORBATE_LIST[i][j]=='Pb':
                    sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_monodentate(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,phi=phi,r=r,attach_atm_ids=ids,offset=offset,pb_id=SORBATE_id,O_id=O_id,mirror=VARS['MIRROR'])           
                elif SORBATE_LIST[i][j]=='Sb':
                    sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_octahedral_monodentate(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,r=r,attach_atm_id=ids,offset=offset,sb_id=SORBATE_id,O_id=O_id)           
                SORBATE_coors_a.append(sorbate_coors[0])
                [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                #now put on sorbate on the symmetrically related domain
                sorbate_ids=[SORBATE_id_B]+O_id_B
                sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            elif len(VARS['SORBATE_ATTACH_ATOM'][i][j])==2:#bidentate case
                top_angle=getattr(VARS['rgh_domain'+str(int(0+1))],'top_angle')
                phi=getattr(VARS['rgh_domain'+str(int(0+1))],'phi')
                ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A']
                offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
                anchor,anchor_offset=None,None
                if ANCHOR_REFERENCE[i][j]!=None:
                    anchor=ANCHOR_REFERENCE[i][j]+'_D'+str(int(i+1))+'A'
                    anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]
                SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                #O_index=[0]+[sum(VARS['O_NUMBER'][i][0:ii+1]) for ii in range(len(VARS['O_NUMBER'][i]))]
                #O_id=VARS['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]
                O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                sorbate_coors=[]
                if SORBATE_LIST[i][j]=='Pb':
                    sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_distortion_B(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,phi=phi,edge_offset=[0,0],attach_atm_ids=ids,offset=offset,anchor_ref=anchor,anchor_offset=anchor_offset,pb_id=SORBATE_id,O_id=O_id,mirror=VARS['MIRROR'])
                elif SORBATE_LIST[i][j]=='Sb':
                    sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_octahedral(domain=VARS['domain'+str(int(i+1))+'A'],theta=phi,phi=top_angle,attach_atm_ids=ids,offset=offset,sb_id=SORBATE_id,O_id=O_id)
                SORBATE_coors_a.append(sorbate_coors[0])
                #if O_NUMBER[i][j]!=0:
                [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                #now put on sorbate on the symmetrically related domain
                sorbate_ids=[SORBATE_id_B]+O_id_B
                sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            elif len(VARS['SORBATE_ATTACH_ATOM'][i][j])==3:#tridentate case (no oxygen sorbate here considering it is a trigonal pyramid structure)
                if SORBATE_LIST[i][j]=='Pb':
                    top_angle=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle')
                    ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+'_D'+str(int(i+1))+'A']
                    offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
                    SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                    sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_pb_share_triple4(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,pb_id=SORBATE_id)
                    SORBATE_coors_a.append(sorbate_coors)
                    SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                    #now put on sorbate on the symmetrically related domain
                    sorbate_ids=[SORBATE_id_B]
                    sorbate_els=[SORBATE_LIST[i][j]]
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
                elif SORBATE_LIST[i][j]=='Sb':
                    ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+'_D'+str(int(i+1))+'A']
                    offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
                    SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                    #O_index=[0]+[sum(VARS['O_NUMBER'][i][0:ii+1]) for ii in range(len(VARS['O_NUMBER'][i]))]
                    #O_id=VARS['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]
                    O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                    sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_share_triple_octahedra(domain=VARS['domain'+str(int(i+1))+'A'],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_oxygen_ids=O_id)
                    SORBATE_coors_a.append(sorbate_coors[0])
                    [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                    SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                    O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                    #now put on sorbate on the symmetrically related domain
                    sorbate_ids=[SORBATE_id_B]+O_id_B
                    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)    

        if USE_BV and i in DOMAINS_BV:
            #set up dynamic super cells,where water and sorbate is a library and surface is a domain instance
            def _widen_validness(value):#acceptable bond valence offset can be adjusted (here is 0.08)
                if value<-BV_TOLERANCE:return 100
                elif value>=-BV_TOLERANCE and value<BV_TOLERANCE:return 0
                else:return value
            def _widen_validness_range(value_min,value_max):#consider a range of (ideal_bv-temp_bv)
                if (value_min<-BV_TOLERANCE and value_max>BV_TOLERANCE) or (value_min>=-BV_TOLERANCE and value_min<=BV_TOLERANCE) or (value_max>=-BV_TOLERANCE and value_max<=BV_TOLERANCE):
                    return 0
                elif value_min>BV_TOLERANCE:return value_min
                else:return 100
            def _widen_validness_hydrogen_acceptor(value,H_N=0):#here consider possible contribution of hydrogen bond (~0.2)
                if (value-H_N*0.2)<-BV_TOLERANCE:return 100
                elif (value-H_N*0.2)>=-BV_TOLERANCE and (value-H_N*0.2)<BV_TOLERANCE:return 0
                else:return (value-H_N*0.2)
            def _widen_validness_potential_hydrogen_acceptor(value):#value=2-temp_bv(temp_bv include covalent hydrogen bond possibly)
                if value<0.2 and value>-BV_TOLERANCE: return 0
                elif value<-BV_TOLERANCE: return 100
                else:return value               
                
            super_cell_water,super_cell_sorbate,super_cell_surface=None,None,None
            if WATER_NUMBER[i]!=0:
                super_cell_water=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+range(-(WATER_NUMBER[i]+sum([np.sum(N_list) for N_list in O_NUMBER[i]])),0))
            if DOMAIN[i]==1:
                super_cell_sorbate=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+range(4,8)+range(-(sum(SORBATE_NUMBER[i])+sum([np.sum(N_list) for N_list in O_NUMBER[i]])+WATER_NUMBER[i]),0))
                super_cell_surface=VARS['domain'+str(i+1)+'A'].copy()
                #delete the first iron layer atoms if considering a half layer
                super_cell_surface.del_atom(super_cell_surface.id[2])
                super_cell_surface.del_atom(super_cell_surface.id[2])
            elif DOMAIN[i]==2:
                super_cell_sorbate=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+range(2,6)+range(-(sum(SORBATE_NUMBER[i])+sum([np.sum(N_list) for N_list in O_NUMBER[i]])+WATER_NUMBER[i]),0))
                super_cell_surface=VARS['domain'+str(i+1)+'A'].copy()
            
            #consider hydrogen bond only among interfacial water molecules and top surface Oxygen layer and Oxygen ligand
            if WATER_NUMBER[i]!=0:
                water_ids=VARS['Os_list_domain'+str(int(i+1))+'a']
                for id in water_ids:
                    tmp_bv=domain_class_1.cal_hydrogen_bond_valence2B(super_cell_water,id,3.,2.5)
                    bv=bv+tmp_bv
                    if debug_bv:bv_container[id]=tmp_bv
            
            #cal bv for surface atoms and sorbates
            #only consdier domainA since domain B is symmetry related to domainA
            for key in VARS['match_lib_'+str(i+1)+'A'].keys():
                temp_bv=None
                if ([sorbate not in key for sorbate in SORBATE]==[True]*len(SORBATE)) and ("HO" not in key):
                    #no searching in this algorithem
                    temp_bv=domain_class_1.cal_bond_valence4B(super_cell_surface,key,VARS['match_lib_'+str(i+1)+'A'][key])
                else:
                    #searching included in this algorithem
                    el=None
                    if "HO" in key:el="O"
                    else:
                        if 'Pb' in key:el='Pb'
                        elif 'Sb' in key:el='Sb'
                    temp_bv=domain_class_1.cal_bond_valence1_new2B(super_cell_sorbate,key,el,2.5,VARS['match_lib_'+str(i+1)+'A'][key],50,False)['total_valence']
                #print key, temp_bv
                #consider possible hydrogen bond and hydroxyl bond fro oxygen atoms
                if 'O' in key:
                    #For O you may consider possible binding to proton (+0.8) 
                    #And note the maximum coordination number for O is 4
                    case_tag=len(VARS['match_lib_'+str(i+1)+'A'][key])
                    if key in map(lambda x:x+'_D'+str(i+1)+'A',COVALENT_HYDROGEN_ACCEPTOR[i]):
                        #if consider convalent hydrogen bond (bv=0.68 to 0.88) while the hydrogen bond has bv from 0.13 to 0.25
                        C_H_N=COVALENT_HYDROGEN_NUMBER[i][map(lambda x:x+'_D'+str(i+1)+'A',COVALENT_HYDROGEN_ACCEPTOR[i]).index(key)]
                        if key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_HYDROGEN_ACCEPTOR[i]):#consider potential hydrogen bond (you can have or have not H-bonding)
                            if _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==0 or _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==100:
                            #if saturated already or over-saturated, then adding H-bonding wont help decrease the the total bv anyhow
                                bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                                if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                            else:
                            #if undersaturation, then compare the cases of inclusion of H-bonding and exclusion of H-bonding. Whichever give rise to the lower bv will be used.
                                bv=bv+min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                                if debug_bv:bv_container[key]=min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                        else:
                            bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                            if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                    else:
                        if key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_HYDROGEN_ACCEPTOR[i]):#consider hydrogen bond
                            if _widen_validness(2-temp_bv)==0 or _widen_validness(2-temp_bv)==100:
                                bv=bv+_widen_validness(2-temp_bv)
                                if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
                            else:
                                bv=bv+min([_widen_validness(2-temp_bv),_widen_validness_range(2-temp_bv-0.25,2-temp_bv-0.13)])
                                if debug_bv:bv_container[key]=min([_widen_validness(2-temp_bv),_widen_validness_range(2-temp_bv-0.25,2-temp_bv-0.13)])
                        else:
                            bv=bv+_widen_validness(2-temp_bv)
                            if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
                elif 'Fe' in key:
                    bv=bv+_widen_validness(3-temp_bv)
                    if debug_bv:bv_container[key]=_widen_validness(3-temp_bv)
                elif ('Pb' in key) or ('Sb' in key):
                    bv=bv+_widen_validness(BV_SUM[i][j]-temp_bv)
                    if debug_bv:bv_container[key]=_widen_validness(BV_SUM[i][j]-temp_bv)
    if debug_bv:
        for i in bv_container.keys():
            print i,bv_container[i]
    #set up multiple domains
    #note for each domain there are two sub domains which symmetrically related to each other, so have equivalent wt
    for i in range(DOMAIN_NUMBER):
        domain['domain'+str(int(i+1))+'A']={'slab':VARS['domain'+str(int(i+1))+'A'],'wt':0.5*vars()['wt_domain'+str(int(i+1))]/total_wt}
        domain['domain'+str(int(i+1))+'B']={'slab':VARS['domain'+str(int(i+1))+'B'],'wt':0.5*vars()['wt_domain'+str(int(i+1))]/total_wt}
      
    if COUNT_TIME:t_2=datetime.now()
    
    #cal structure factor for each dataset in this for loop
    for data_set in data:
        f=np.array([])   
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        x = data_set.x
        y = data_set.extra_data['Y']
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']
        if x[0]>100:#a sign for RAXS dataset(first column is Energy which is in the order of 1000 ev)
            sample = model2.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.1391})
            rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
            f = SCALES[1]*rough*sample.calc_f(h, k, y,f1f2,res_el)
            F.append(abs(f))
            fom_scaler.append(WT_RAXS)
        else:#First column is l for CTR dataset, l is a relative small number (less than 10 usually)
            sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.1391})
            rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(x-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
            f = SCALES[0]*rough*sample.calc_f3(h, k, x)
            if h[0]==0 and k[0]==0:#extra scale factor for specular rod
                f=SCALES[2]*f
            F.append(abs(f))
            fom_scaler.append(1)
    #print domain_creator.extract_coor(domain1B,'Pb1_D1B')
    #print domain_creator.extract_component(domain1A,'O1_1_0_D1A',['dx1','dy2','dz3'])  
    #domain_creator.print_data(N_sorbate=2,N_atm=40,domain=domain1A,z_shift=1,half_layer=True,save_file='D://model2.xyz')    
    #export the model results for plotting if PLOT set to true
    #domain_creator.layer_spacing_calculator(domain1A,12,True)
    #print domain_class_1.cal_bond_valence1(domain_class_1.build_super_cell2(domain1A,[0,1,4,5]+range(-6,0)),'Pb1_D1A',3,False)
    #print domain_class_1.cal_bond_valence1(domain_class_1.build_super_cell(domain1A),'O1_6_0_D1A',2.5,False)
    #print domain_class_1.cal_bond_valence1(domain_class_1.build_super_cell2(domain1A,[0,1,4,5]+range(-6,0)),'Pb1_D1A',3,False)
    #print domain_class_1.cal_bond_valence1_new2(domain_class_1.build_super_cell2(domain1A,[0,1,4,5]+range(-6,0)),'Pb1_D1A',3,['HO1_D1','O1_1_0','O1_2_0'],10,False)

    if PLOT:
        bl_dl={'3_0':{'segment':[[0,1],[1,8]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,8]],'info':[[2,2.0]]},'2_1':{'segment':[[0,8]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,8]],'info':[[2,1.7218]]},\
            '2_-1':{'segment':[[0,3.1391],[3.1391,8]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,8]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,8]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,8]],'info':[[2,1.7218]]},\
            '0_0':{'segment':[[0,13]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,8]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,8]],'info':[[2,-6.2782]]},\
            '-2_-2':{'segment':[[0,8]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,8]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,8]],'info':[[2,-6]]},\
            '-2_1':{'segment':[[0,4.8609],[4.8609,8]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,8]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,8]],'info':[[2,-1],[6,-1]]}}

        plot_data_container_experiment={}
        plot_data_container_model={}
        for data_set in data:
            f=np.array([])   
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            I=data_set.y
            eI=data_set.error
            #make dumy hkl and f to make the plot look smoother
            l_dumy=np.arange(l[0],l[-1],0.1)
            N=len(l_dumy)
            h_dumy=np.array([h[0]]*N)
            k_dumy=np.array([k[0]]*N)
            LB_dumy=[]
            dL_dumy=[]
            
            for i in range(N):
                key=None
                if l_dumy[i]>=0:
                    key=str(int(h[0]))+'_'+str(int(k[0]))
                else:key=str(int(-h[0]))+'_'+str(int(-k[0]))
                for ii in bl_dl[key]['segment']:
                    if abs(l_dumy[i])>=ii[0] and abs(l_dumy[i])<ii[1]:
                        n=bl_dl[key]['segment'].index(ii)
                        LB_dumy.append(bl_dl[key]['info'][n][1])
                        dL_dumy.append(bl_dl[key]['info'][n][0])
            LB_dumy=np.array(LB_dumy)
            dL_dumy=np.array(dL_dumy)
            rough_dumy = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l_dumy-LB_dumy)/dL_dumy)**2)**0.5
            f_dumy = SCALES[0]*rough_dumy*sample.calc_f(h_dumy, k_dumy, l_dumy)
            
            label=str(int(h[0]))+str(int(k[0]))+'L'
            plot_data_container_experiment[label]=np.concatenate((l[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
            plot_data_container_model[label]=np.concatenate((l_dumy[:,np.newaxis],f_dumy[:,np.newaxis]),axis=1)
        hkls=['00L','02L','10L','11L','20L','22L','30L','2-1L','21L']
        plot_data_list=[]
        for hkl in hkls:
            plot_data_list.append([plot_data_container_experiment[hkl],plot_data_container_model[hkl]])
        pickle.dump(plot_data_list,open("D:\\Google Drive\\useful codes\\plotting\\temp_plot","wb"))
    #you may play with the weighting rule by setting eg 2**bv, 5**bv for the wt factor, that way you are pushing the GenX to find a fit btween 
    #good fit (low wt factor) and a reasonable fit (high wt factor)
    if COUNT_TIME:t_3=datetime.now()
    if COUNT_TIME:
        print "It took "+str(t_1-t_0)+" seconds to setup"
        print "It took "+str(t_2-t_1)+" seconds to calculate bv weighting"
        print "It took "+str(t_3-t_2)+" seconds to calculate structure factor"
    return F,1+WT_BV*bv,fom_scaler
    
#############################################code update logs############################################################################
"""
##version 1##
consider sorbate (Pb and Sb) of any combination
"""

#########################naming rules###################
"""
    ###########atm ids################
    sorbate: Pb1_D1A, Pb2_D2B, HO1_Pb1_D1A,HO2_Pb1_D1A, HO1_Sb2_D1A
    water: Os1_D1A
    surf atms: O1_1_0_D1A (half layer), O1_1_1_D1A(atoms of second slab)
    ############group names###########
    gp_Pb1_D1(discrete grouping for sorbate, group u dx dy dz)
    gp_O1O7_D1(discrete grouping for surface atms, group dx dy in symmetry)
    gp_sorbates_set1_D1(discrete grouping for each set of sorbates (O and metal), group oc)
    gp_HO_set1_D1(discrete grouping for each set of oxygen sorbates, group u)
    gp_waters_set1_D1(discrete grouping for each set of water at same layer, group u, oc and dz)
    gp_O1O2_O7O8_D1(sequence grouping for u, oc and dz)

    some print examples
    #print domain_creator.extract_coor(domain1A,'O1_1_0_D1A')
    #print domain_creator.extract_coor(domain1B,'Pb1_D1B')
    #print_data(N_sorbate=4,N_atm=40,domain=domain1A,z_shift=1,save_file='D://model.xyz')
    #print domain_class_1.cal_bond_valence1(domain1A,'Pb1_D1A',3,False)
"""

###########explanation of some global variables###################
#pars for interfacial waters##
"""
    WATER_NUMBER: must be even number considering 2 atoms each layer
    V_SHIFT: vertical shiftment of water molecules, in unit of angstroms,two items each if consider two water pair
    R=: half distance bw two waters at each layer in unit of angstroms
    ALPHA: alpha angle used to cal the pos of water molecule
    
    DISCONNECT_BV_CONTRIBUTION=[{('O1_1_0','O1_2_0'):'Pb2'},{}]
    ##if you have two sorbate within the same unit cell, two sorbates will be the coordinative ligands of anchor atoms
    ##but in fact the sorbate cannot occupy the adjacent sites due to steric constrain, so you should delete one ligand in the case of average structure
    ##However, if you consider multiple domains with each domain having only one sorbate, then set the items to be {}
    ##in this case, bv contribution from Pb2 won't be account for both O1 and O2 atom.
        
    ANCHOR_REFERENCE=[[None],[None]]
    ANCHOR_REFERENCE_OFFSET=[[None],[None]]
    #we use anchor reference to set up a binding configuration in a more intelligible way. We only specify the anchor ref when the two anchor points are 
    #not on the same level. The anchor reference will be the center(Fe atom) of the octahedral coordinated by ligands including those two anchors.
    #phi=0 will means that the sorbate is binded in a most feasible way.
    #To ensure the bending on two symmetry site are towards right position, the sorbate attach atom may have reversed order.
    #eg. [O1,O3] correspond to [O4px,O2] rather than [O2,O4px].
    
    COVALENT_HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0'],['O1_1_0','O1_2_0']]
    COVALENT_HYDROGEN_NUMBER=[[1,1],[1,1]]
    ##means in domain1 both O1 and O2 will accept one covalent hydrogen (bv contribution of 0.8)
    
    HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0'],['O1_1_0','O1_2_0']]
    HYDROGEN_NUMBER=[[1,1,1,1],[1,1]]
    ##means in domain1 O1 to O4 will accept one covalent hygrogen (bv contribution of 0.2)
"""

##########################some examples to set up sorbates binding under different configurations#######################
"""
SORBATE=["Pb","Sb"]
eg1.SORBATE_NUMBER=[[1,0],[0,1]]#two domains:one has one Pb sorbate and the other one has one Sb sorbate
eg2.SORBATE_NUMBER=[[0,0],[0,1]]#two domains:first one is clean surface, the other one has one Sb sorbate
eg3.SORBATE_NUMBER=[[1,1],[1,1]]#two domains and each domain has one Pb and one Sb sorbate
eg4.SORBATE_NUMBER=[[2,0],[0,1]]#first domain has two Pb sorbate

#len(O_NUMBER)= # of domains
#len(O_NUMBER[i])= # of sorbates
O_NUMBER=[[[1],[0]],[[0],[0]]]#Pb sorbate has one oxygen ligand in domain1, while no oxygen ligands in domain2
O_NUMBER=[[[1,2],[0]],[[0],[3,5]]]#1st Pb sorbate has one oxygen ligand and 2nd Pb has two oxygen ligands in domain1, while in domain2 1st Sb has 3 oxygen ligands and 2nd has 5 oxygen ligands
SORBATE_LIST=create_sorbate_el_list(SORBATE,SORBATE_NUMBER)
BV_SUM=[[1.33,5],[1.33,5.]]#pseudo bond valence sum for sorbate

#len(SORBATE_ATTACH_ATOM)=# of domains
#len(SORBATE_ATTACH_ATOM[i])=# of sorbates in domaini
SORBATE_ATTACH_ATOM=[[['O1_1_0','O1_2_0']],[['O1_1_0','O1_2_0','O1_3_0']]]
SORBATE_ATTACH_ATOM_OFFSET=[[[None,None]],[[None,None,None]]]
TOP_ANGLE=[[1.38],[1.38]]
PHI=[[0],[0]]
R_S=[[1],[1]]
MIRROR=False
"""
##########################offset symbols######################
"""
None: no translation
'+x':translate along positive x axis by 1 unit
'-x':translate along negative x axis by 1 unit
SAME deal for translation along y axis
"""