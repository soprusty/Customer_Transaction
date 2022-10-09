import sklearn.neighbors as knn
import streamlit as st
import pandas as pd
import pickle

#Loading up the Regression model we created
model = knn.KNeighborsClassifier()
#model.load_model('finalized_model.sav')

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

#Caching the model for faster loading
@st.cache


# Define the prediction function
def predict(var_0,
var_1	,
var_2	,
var_3	,
var_4	,
var_5	,
var_6	,
var_7	,
var_8	,
var_9	,
var_10	,
var_11	,
var_12	,
var_13	,
var_14	,
var_15	,
var_16	,
var_17	,
var_18	,
var_19	,
var_20	,
var_21	,
var_22	,
var_23	,
var_24	,
var_25	,
var_26	,
var_27	,
var_28	,
var_29	,
var_30	,
var_31	,
var_32	,
var_33	,
var_34	,
var_35	,
var_36	,
var_37	,
var_38	,
var_39	,
var_40	,
var_41	,
var_42	,
var_43	,
var_44	,
var_45	,
var_46	,
var_47	,
var_48	,
var_49	,
var_50	,
var_51	,
var_52	,
var_53	,
var_54	,
var_55	,
var_56	,
var_57	,
var_58	,
var_59	,
var_60	,
var_61	,
var_62	,
var_63	,
var_64	,
var_65	,
var_66	,
var_67	,
var_68	,
var_69	,
var_70	,
var_71	,
var_72	,
var_73	,
var_74	,
var_75	,
var_76	,
var_77	,
var_78	,
var_79	,
var_80	,
var_81	,
var_82	,
var_83	,
var_84	,
var_85	,
var_86	,
var_87	,
var_88	,
var_89	,
var_90	,
var_91	,
var_92	,
var_93	,
var_94	,
var_95	,
var_96	,
var_97	,
var_98	,
var_99	,
var_100	,
var_101	,
var_102	,
var_103	,
var_104	,
var_105	,
var_106	,
var_107	,
var_108	,
var_109	,
var_110	,
var_111	,
var_112	,
var_113	,
var_114	,
var_115	,
var_116	,
var_117	,
var_118	,
var_119	,
var_120	,
var_121	,
var_122	,
var_123	,
var_124	,
var_125	,
var_126	,
var_127	,
var_128	,
var_129	,
var_130	,
var_131	,
var_132	,
var_133	,
var_134	,
var_135	,
var_136	,
var_137	,
var_138	,
var_139	,
var_140	,
var_141	,
var_142	,
var_143	,
var_144	,
var_145	,
var_146	,
var_147	,
var_148	,
var_149	,
var_150	,
var_151	,
var_152	,
var_153	,
var_154	,
var_155	,
var_156	,
var_157	,
var_158	,
var_159	,
var_160	,
var_161	,
var_162	,
var_163	,
var_164	,
var_165	,
var_166	,
var_167	,
var_168	,
var_169	,
var_170	,
var_171	,
var_172	,
var_173	,
var_174	,
var_175	,
var_176	,
var_177	,
var_178	,
var_179	,
var_180	,
var_181	,
var_182	,
var_183	,
var_184	,
var_185	,
var_186	,
var_187	,
var_188	,
var_189	,
var_190	,
var_191	,
var_192	,
var_193	,
var_194	,
var_195	,
var_196	,
var_197	,
var_198	,
var_199):
    
    
    

    prediction = loaded_model.predict(pd.DataFrame([[var_0	,
var_1	,
var_2	,
var_3	,
var_4	,
var_5	,
var_6	,
var_7	,
var_8	,
var_9	,
var_10	,
var_11	,
var_12	,
var_13	,
var_14	,
var_15	,
var_16	,
var_17	,
var_18	,
var_19	,
var_20	,
var_21	,
var_22	,
var_23	,
var_24	,
var_25	,
var_26	,
var_27	,
var_28	,
var_29	,
var_30	,
var_31	,
var_32	,
var_33	,
var_34	,
var_35	,
var_36	,
var_37	,
var_38	,
var_39	,
var_40	,
var_41	,
var_42	,
var_43	,
var_44	,
var_45	,
var_46	,
var_47	,
var_48	,
var_49	,
var_50	,
var_51	,
var_52	,
var_53	,
var_54	,
var_55	,
var_56	,
var_57	,
var_58	,
var_59	,
var_60	,
var_61	,
var_62	,
var_63	,
var_64	,
var_65	,
var_66	,
var_67	,
var_68	,
var_69	,
var_70	,
var_71	,
var_72	,
var_73	,
var_74	,
var_75	,
var_76	,
var_77	,
var_78	,
var_79	,
var_80	,
var_81	,
var_82	,
var_83	,
var_84	,
var_85	,
var_86	,
var_87	,
var_88	,
var_89	,
var_90	,
var_91	,
var_92	,
var_93	,
var_94	,
var_95	,
var_96	,
var_97	,
var_98	,
var_99	,
var_100	,
var_101	,
var_102	,
var_103	,
var_104	,
var_105	,
var_106	,
var_107	,
var_108	,
var_109	,
var_110	,
var_111	,
var_112	,
var_113	,
var_114	,
var_115	,
var_116	,
var_117	,
var_118	,
var_119	,
var_120	,
var_121	,
var_122	,
var_123	,
var_124	,
var_125	,
var_126	,
var_127	,
var_128	,
var_129	,
var_130	,
var_131	,
var_132	,
var_133	,
var_134	,
var_135	,
var_136	,
var_137	,
var_138	,
var_139	,
var_140	,
var_141	,
var_142	,
var_143	,
var_144	,
var_145	,
var_146	,
var_147	,
var_148	,
var_149	,
var_150	,
var_151	,
var_152	,
var_153	,
var_154	,
var_155	,
var_156	,
var_157	,
var_158	,
var_159	,
var_160	,
var_161	,
var_162	,
var_163	,
var_164	,
var_165	,
var_166	,
var_167	,
var_168	,
var_169	,
var_170	,
var_171	,
var_172	,
var_173	,
var_174	,
var_175	,
var_176	,
var_177	,
var_178	,
var_179	,
var_180	,
var_181	,
var_182	,
var_183	,
var_184	,
var_185	,
var_186	,
var_187	,
var_188	,
var_189	,
var_190	,
var_191	,
var_192	,
var_193	,
var_194	,
var_195	,
var_196	,
var_197	,
var_198	,
var_199]], columns=['var_0',
'var_1',
'var_2',
'var_3',
'var_4',
'var_5',
'var_6',
'var_7',
'var_8',
'var_9',
'var_10',
'var_11',
'var_12',
'var_13',
'var_14',
'var_15',
'var_16',
'var_17',
'var_18',
'var_19',
'var_20',
'var_21',
'var_22',
'var_23',
'var_24',
'var_25',
'var_26',
'var_27',
'var_28',
'var_29',
'var_30',
'var_31',
'var_32',
'var_33',
'var_34',
'var_35',
'var_36',
'var_37',
'var_38',
'var_39',
'var_40',
'var_41',
'var_42',
'var_43',
'var_44',
'var_45',
'var_46',
'var_47',
'var_48',
'var_49',
'var_50',
'var_51',
'var_52',
'var_53',
'var_54',
'var_55',
'var_56',
'var_57',
'var_58',
'var_59',
'var_60',
'var_61',
'var_62',
'var_63',
'var_64',
'var_65',
'var_66',
'var_67',
'var_68',
'var_69',
'var_70',
'var_71',
'var_72',
'var_73',
'var_74',
'var_75',
'var_76',
'var_77',
'var_78',
'var_79',
'var_80',
'var_81',
'var_82',
'var_83',
'var_84',
'var_85',
'var_86',
'var_87',
'var_88',
'var_89',
'var_90',
'var_91',
'var_92',
'var_93',
'var_94',
'var_95',
'var_96',
'var_97',
'var_98',
'var_99',
'var_100',
'var_101',
'var_102',
'var_103',
'var_104',
'var_105',
'var_106',
'var_107',
'var_108',
'var_109',
'var_110',
'var_111',
'var_112',
'var_113',
'var_114',
'var_115',
'var_116',
'var_117',
'var_118',
'var_119',
'var_120',
'var_121',
'var_122',
'var_123',
'var_124',
'var_125',
'var_126',
'var_127',
'var_128',
'var_129',
'var_130',
'var_131',
'var_132',
'var_133',
'var_134',
'var_135',
'var_136',
'var_137',
'var_138',
'var_139',
'var_140',
'var_141',
'var_142',
'var_143',
'var_144',
'var_145',
'var_146',
'var_147',
'var_148',
'var_149',
'var_150',
'var_151',
'var_152',
'var_153',
'var_154',
'var_155',
'var_156',
'var_157',
'var_158',
'var_159',
'var_160',
'var_161',
'var_162',
'var_163',
'var_164',
'var_165',
'var_166',
'var_167',
'var_168',
'var_169',
'var_170',
'var_171',
'var_172',
'var_173',
'var_174',
'var_175',
'var_176',
'var_177',
'var_178',
'var_179',
'var_180',
'var_181',
'var_182',
'var_183',
'var_184',
'var_185',
'var_186',
'var_187',
'var_188',
'var_189',
'var_190',
'var_191',
'var_192',
'var_193',
'var_194',
'var_195',
'var_196',
'var_197',
'var_198',
'var_199']))
    return prediction


st.title('Customer Transaction Predictor')
st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")
st.header('Enter the characteristics of the Customer:')
#carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)
#cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
#color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
#clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
#depth = st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=1.0)
#table = st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=1.0)
#x = st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=1.0)
#y = st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=1.0)
#z = st.number_input('Diamond Height (Z) in mm:', min_value=0.1, max_value=100.0, value=1.0)

var_0 = st.number_input('var_0:',min_value=0.1, max_value=100.0, value=1.0)
var_1 = st.number_input('var_1:',min_value=0.1, max_value=100.0, value=1.0)
var_2 = st.number_input('var_2:',min_value=0.1, max_value=100.0, value=1.0)
var_3 = st.number_input('var_3:',min_value=0.1, max_value=100.0, value=1.0)
var_4 = st.number_input('var_4:',min_value=0.1, max_value=100.0, value=1.0)
var_5 = st.number_input('var_5:',min_value=0.1, max_value=100.0, value=1.0)
var_6 = st.number_input('var_6:',min_value=0.1, max_value=100.0, value=1.0)
var_7 = st.number_input('var_7:',min_value=0.1, max_value=100.0, value=1.0)
var_8 = st.number_input('var_8:',min_value=0.1, max_value=100.0, value=1.0)
var_9 = st.number_input('var_9:',min_value=0.1, max_value=100.0, value=1.0)
var_10 = st.number_input('var_10:',min_value=0.1, max_value=100.0, value=1.0)
var_11 = st.number_input('var_11:',min_value=0.1, max_value=100.0, value=1.0)
var_12 = st.number_input('var_12:',min_value=0.1, max_value=100.0, value=1.0)
var_13 = st.number_input('var_13:',min_value=0.1, max_value=100.0, value=1.0)
var_14 = st.number_input('var_14:',min_value=0.1, max_value=100.0, value=1.0)
var_15 = st.number_input('var_15:',min_value=0.1, max_value=100.0, value=1.0)
var_16 = st.number_input('var_16:',min_value=0.1, max_value=100.0, value=1.0)
var_17 = st.number_input('var_17:',min_value=0.1, max_value=100.0, value=1.0)
var_18 = st.number_input('var_18:',min_value=0.1, max_value=100.0, value=1.0)
var_19 = st.number_input('var_19:',min_value=0.1, max_value=100.0, value=1.0)
var_20 = st.number_input('var_20:',min_value=0.1, max_value=100.0, value=1.0)
var_21 = st.number_input('var_21:',min_value=0.1, max_value=100.0, value=1.0)
var_22 = st.number_input('var_22:',min_value=0.1, max_value=100.0, value=1.0)
var_23 = st.number_input('var_23:',min_value=0.1, max_value=100.0, value=1.0)
var_24 = st.number_input('var_24:',min_value=0.1, max_value=100.0, value=1.0)
var_25 = st.number_input('var_25:',min_value=0.1, max_value=100.0, value=1.0)
var_26 = st.number_input('var_26:',min_value=0.1, max_value=100.0, value=1.0)
var_27 = st.number_input('var_27:',min_value=0.1, max_value=100.0, value=1.0)
var_28 = st.number_input('var_28:',min_value=0.1, max_value=100.0, value=1.0)
var_29 = st.number_input('var_29:',min_value=0.1, max_value=100.0, value=1.0)
var_30 = st.number_input('var_30:',min_value=0.1, max_value=100.0, value=1.0)
var_31 = st.number_input('var_31:',min_value=0.1, max_value=100.0, value=1.0)
var_32 = st.number_input('var_32:',min_value=0.1, max_value=100.0, value=1.0)
var_33 = st.number_input('var_33:',min_value=0.1, max_value=100.0, value=1.0)
var_34 = st.number_input('var_34:',min_value=0.1, max_value=100.0, value=1.0)
var_35 = st.number_input('var_35:',min_value=0.1, max_value=100.0, value=1.0)
var_36 = st.number_input('var_36:',min_value=0.1, max_value=100.0, value=1.0)
var_37 = st.number_input('var_37:',min_value=0.1, max_value=100.0, value=1.0)
var_38 = st.number_input('var_38:',min_value=0.1, max_value=100.0, value=1.0)
var_39 = st.number_input('var_39:',min_value=0.1, max_value=100.0, value=1.0)
var_40 = st.number_input('var_40:',min_value=0.1, max_value=100.0, value=1.0)
var_41 = st.number_input('var_41:',min_value=0.1, max_value=100.0, value=1.0)
var_42 = st.number_input('var_42:',min_value=0.1, max_value=100.0, value=1.0)
var_43 = st.number_input('var_43:',min_value=0.1, max_value=100.0, value=1.0)
var_44 = st.number_input('var_44:',min_value=0.1, max_value=100.0, value=1.0)
var_45 = st.number_input('var_45:',min_value=0.1, max_value=100.0, value=1.0)
var_46 = st.number_input('var_46:',min_value=0.1, max_value=100.0, value=1.0)
var_47 = st.number_input('var_47:',min_value=0.1, max_value=100.0, value=1.0)
var_48 = st.number_input('var_48:',min_value=0.1, max_value=100.0, value=1.0)
var_49 = st.number_input('var_49:',min_value=0.1, max_value=100.0, value=1.0)
var_50 = st.number_input('var_50:',min_value=0.1, max_value=100.0, value=1.0)
var_51 = st.number_input('var_51:',min_value=0.1, max_value=100.0, value=1.0)
var_52 = st.number_input('var_52:',min_value=0.1, max_value=100.0, value=1.0)
var_53 = st.number_input('var_53:',min_value=0.1, max_value=100.0, value=1.0)
var_54 = st.number_input('var_54:',min_value=0.1, max_value=100.0, value=1.0)
var_55 = st.number_input('var_55:',min_value=0.1, max_value=100.0, value=1.0)
var_56 = st.number_input('var_56:',min_value=0.1, max_value=100.0, value=1.0)
var_57 = st.number_input('var_57:',min_value=0.1, max_value=100.0, value=1.0)
var_58 = st.number_input('var_58:',min_value=0.1, max_value=100.0, value=1.0)
var_59 = st.number_input('var_59:',min_value=0.1, max_value=100.0, value=1.0)
var_60 = st.number_input('var_60:',min_value=0.1, max_value=100.0, value=1.0)
var_61 = st.number_input('var_61:',min_value=0.1, max_value=100.0, value=1.0)
var_62 = st.number_input('var_62:',min_value=0.1, max_value=100.0, value=1.0)
var_63 = st.number_input('var_63:',min_value=0.1, max_value=100.0, value=1.0)
var_64 = st.number_input('var_64:',min_value=0.1, max_value=100.0, value=1.0)
var_65 = st.number_input('var_65:',min_value=0.1, max_value=100.0, value=1.0)
var_66 = st.number_input('var_66:',min_value=0.1, max_value=100.0, value=1.0)
var_67 = st.number_input('var_67:',min_value=0.1, max_value=100.0, value=1.0)
var_68 = st.number_input('var_68:',min_value=0.1, max_value=100.0, value=1.0)
var_69 = st.number_input('var_69:',min_value=0.1, max_value=100.0, value=1.0)
var_70 = st.number_input('var_70:',min_value=0.1, max_value=100.0, value=1.0)
var_71 = st.number_input('var_71:',min_value=0.1, max_value=100.0, value=1.0)
var_72 = st.number_input('var_72:',min_value=0.1, max_value=100.0, value=1.0)
var_73 = st.number_input('var_73:',min_value=0.1, max_value=100.0, value=1.0)
var_74 = st.number_input('var_74:',min_value=0.1, max_value=100.0, value=1.0)
var_75 = st.number_input('var_75:',min_value=0.1, max_value=100.0, value=1.0)
var_76 = st.number_input('var_76:',min_value=0.1, max_value=100.0, value=1.0)
var_77 = st.number_input('var_77:',min_value=0.1, max_value=100.0, value=1.0)
var_78 = st.number_input('var_78:',min_value=0.1, max_value=100.0, value=1.0)
var_79 = st.number_input('var_79:',min_value=0.1, max_value=100.0, value=1.0)
var_80 = st.number_input('var_80:',min_value=0.1, max_value=100.0, value=1.0)
var_81 = st.number_input('var_81:',min_value=0.1, max_value=100.0, value=1.0)
var_82 = st.number_input('var_82:',min_value=0.1, max_value=100.0, value=1.0)
var_83 = st.number_input('var_83:',min_value=0.1, max_value=100.0, value=1.0)
var_84 = st.number_input('var_84:',min_value=0.1, max_value=100.0, value=1.0)
var_85 = st.number_input('var_85:',min_value=0.1, max_value=100.0, value=1.0)
var_86 = st.number_input('var_86:',min_value=0.1, max_value=100.0, value=1.0)
var_87 = st.number_input('var_87:',min_value=0.1, max_value=100.0, value=1.0)
var_88 = st.number_input('var_88:',min_value=0.1, max_value=100.0, value=1.0)
var_89 = st.number_input('var_89:',min_value=0.1, max_value=100.0, value=1.0)
var_90 = st.number_input('var_90:',min_value=0.1, max_value=100.0, value=1.0)
var_91 = st.number_input('var_91:',min_value=0.1, max_value=100.0, value=1.0)
var_92 = st.number_input('var_92:',min_value=0.1, max_value=100.0, value=1.0)
var_93 = st.number_input('var_93:',min_value=0.1, max_value=100.0, value=1.0)
var_94 = st.number_input('var_94:',min_value=0.1, max_value=100.0, value=1.0)
var_95 = st.number_input('var_95:',min_value=0.1, max_value=100.0, value=1.0)
var_96 = st.number_input('var_96:',min_value=0.1, max_value=100.0, value=1.0)
var_97 = st.number_input('var_97:',min_value=0.1, max_value=100.0, value=1.0)
var_98 = st.number_input('var_98:',min_value=0.1, max_value=100.0, value=1.0)
var_99 = st.number_input('var_99:',min_value=0.1, max_value=100.0, value=1.0)
var_100 = st.number_input('var_100:',min_value=0.1, max_value=100.0, value=1.0)
var_101 = st.number_input('var_101:',min_value=0.1, max_value=100.0, value=1.0)
var_102 = st.number_input('var_102:',min_value=0.1, max_value=100.0, value=1.0)
var_103 = st.number_input('var_103:',min_value=0.1, max_value=100.0, value=1.0)
var_104 = st.number_input('var_104:',min_value=0.1, max_value=100.0, value=1.0)
var_105 = st.number_input('var_105:',min_value=0.1, max_value=100.0, value=1.0)
var_106 = st.number_input('var_106:',min_value=0.1, max_value=100.0, value=1.0)
var_107 = st.number_input('var_107:',min_value=0.1, max_value=100.0, value=1.0)
var_108 = st.number_input('var_108:',min_value=0.1, max_value=100.0, value=1.0)
var_109 = st.number_input('var_109:',min_value=0.1, max_value=100.0, value=1.0)
var_110 = st.number_input('var_110:',min_value=0.1, max_value=100.0, value=1.0)
var_111 = st.number_input('var_111:',min_value=0.1, max_value=100.0, value=1.0)
var_112 = st.number_input('var_112:',min_value=0.1, max_value=100.0, value=1.0)
var_113 = st.number_input('var_113:',min_value=0.1, max_value=100.0, value=1.0)
var_114 = st.number_input('var_114:',min_value=0.1, max_value=100.0, value=1.0)
var_115 = st.number_input('var_115:',min_value=0.1, max_value=100.0, value=1.0)
var_116 = st.number_input('var_116:',min_value=0.1, max_value=100.0, value=1.0)
var_117 = st.number_input('var_117:',min_value=0.1, max_value=100.0, value=1.0)
var_118 = st.number_input('var_118:',min_value=0.1, max_value=100.0, value=1.0)
var_119 = st.number_input('var_119:',min_value=0.1, max_value=100.0, value=1.0)
var_120 = st.number_input('var_120:',min_value=0.1, max_value=100.0, value=1.0)
var_121 = st.number_input('var_121:',min_value=0.1, max_value=100.0, value=1.0)
var_122 = st.number_input('var_122:',min_value=0.1, max_value=100.0, value=1.0)
var_123 = st.number_input('var_123:',min_value=0.1, max_value=100.0, value=1.0)
var_124 = st.number_input('var_124:',min_value=0.1, max_value=100.0, value=1.0)
var_125 = st.number_input('var_125:',min_value=0.1, max_value=100.0, value=1.0)
var_126 = st.number_input('var_126:',min_value=0.1, max_value=100.0, value=1.0)
var_127 = st.number_input('var_127:',min_value=0.1, max_value=100.0, value=1.0)
var_128 = st.number_input('var_128:',min_value=0.1, max_value=100.0, value=1.0)
var_129 = st.number_input('var_129:',min_value=0.1, max_value=100.0, value=1.0)
var_130 = st.number_input('var_130:',min_value=0.1, max_value=100.0, value=1.0)
var_131 = st.number_input('var_131:',min_value=0.1, max_value=100.0, value=1.0)
var_132 = st.number_input('var_132:',min_value=0.1, max_value=100.0, value=1.0)
var_133 = st.number_input('var_133:',min_value=0.1, max_value=100.0, value=1.0)
var_134 = st.number_input('var_134:',min_value=0.1, max_value=100.0, value=1.0)
var_135 = st.number_input('var_135:',min_value=0.1, max_value=100.0, value=1.0)
var_136 = st.number_input('var_136:',min_value=0.1, max_value=100.0, value=1.0)
var_137 = st.number_input('var_137:',min_value=0.1, max_value=100.0, value=1.0)
var_138 = st.number_input('var_138:',min_value=0.1, max_value=100.0, value=1.0)
var_139 = st.number_input('var_139:',min_value=0.1, max_value=100.0, value=1.0)
var_140 = st.number_input('var_140:',min_value=0.1, max_value=100.0, value=1.0)
var_141 = st.number_input('var_141:',min_value=0.1, max_value=100.0, value=1.0)
var_142 = st.number_input('var_142:',min_value=0.1, max_value=100.0, value=1.0)
var_143 = st.number_input('var_143:',min_value=0.1, max_value=100.0, value=1.0)
var_144 = st.number_input('var_144:',min_value=0.1, max_value=100.0, value=1.0)
var_145 = st.number_input('var_145:',min_value=0.1, max_value=100.0, value=1.0)
var_146 = st.number_input('var_146:',min_value=0.1, max_value=100.0, value=1.0)
var_147 = st.number_input('var_147:',min_value=0.1, max_value=100.0, value=1.0)
var_148 = st.number_input('var_148:',min_value=0.1, max_value=100.0, value=1.0)
var_149 = st.number_input('var_149:',min_value=0.1, max_value=100.0, value=1.0)
var_150 = st.number_input('var_150:',min_value=0.1, max_value=100.0, value=1.0)
var_151 = st.number_input('var_151:',min_value=0.1, max_value=100.0, value=1.0)
var_152 = st.number_input('var_152:',min_value=0.1, max_value=100.0, value=1.0)
var_153 = st.number_input('var_153:',min_value=0.1, max_value=100.0, value=1.0)
var_154 = st.number_input('var_154:',min_value=0.1, max_value=100.0, value=1.0)
var_155 = st.number_input('var_155:',min_value=0.1, max_value=100.0, value=1.0)
var_156 = st.number_input('var_156:',min_value=0.1, max_value=100.0, value=1.0)
var_157 = st.number_input('var_157:',min_value=0.1, max_value=100.0, value=1.0)
var_158 = st.number_input('var_158:',min_value=0.1, max_value=100.0, value=1.0)
var_159 = st.number_input('var_159:',min_value=0.1, max_value=100.0, value=1.0)
var_160 = st.number_input('var_160:',min_value=0.1, max_value=100.0, value=1.0)
var_161 = st.number_input('var_161:',min_value=0.1, max_value=100.0, value=1.0)
var_162 = st.number_input('var_162:',min_value=0.1, max_value=100.0, value=1.0)
var_163 = st.number_input('var_163:',min_value=0.1, max_value=100.0, value=1.0)
var_164 = st.number_input('var_164:',min_value=0.1, max_value=100.0, value=1.0)
var_165 = st.number_input('var_165:',min_value=0.1, max_value=100.0, value=1.0)
var_166 = st.number_input('var_166:',min_value=0.1, max_value=100.0, value=1.0)
var_167 = st.number_input('var_167:',min_value=0.1, max_value=100.0, value=1.0)
var_168 = st.number_input('var_168:',min_value=0.1, max_value=100.0, value=1.0)
var_169 = st.number_input('var_169:',min_value=0.1, max_value=100.0, value=1.0)
var_170 = st.number_input('var_170:',min_value=0.1, max_value=100.0, value=1.0)
var_171 = st.number_input('var_171:',min_value=0.1, max_value=100.0, value=1.0)
var_172 = st.number_input('var_172:',min_value=0.1, max_value=100.0, value=1.0)
var_173 = st.number_input('var_173:',min_value=0.1, max_value=100.0, value=1.0)
var_174 = st.number_input('var_174:',min_value=0.1, max_value=100.0, value=1.0)
var_175 = st.number_input('var_175:',min_value=0.1, max_value=100.0, value=1.0)
var_176 = st.number_input('var_176:',min_value=0.1, max_value=100.0, value=1.0)
var_177 = st.number_input('var_177:',min_value=0.1, max_value=100.0, value=1.0)
var_178 = st.number_input('var_178:',min_value=0.1, max_value=100.0, value=1.0)
var_179 = st.number_input('var_179:',min_value=0.1, max_value=100.0, value=1.0)
var_180 = st.number_input('var_180:',min_value=0.1, max_value=100.0, value=1.0)
var_181 = st.number_input('var_181:',min_value=0.1, max_value=100.0, value=1.0)
var_182 = st.number_input('var_182:',min_value=0.1, max_value=100.0, value=1.0)
var_183 = st.number_input('var_183:',min_value=0.1, max_value=100.0, value=1.0)
var_184 = st.number_input('var_184:',min_value=0.1, max_value=100.0, value=1.0)
var_185 = st.number_input('var_185:',min_value=0.1, max_value=100.0, value=1.0)
var_186 = st.number_input('var_186:',min_value=0.1, max_value=100.0, value=1.0)
var_187 = st.number_input('var_187:',min_value=0.1, max_value=100.0, value=1.0)
var_188 = st.number_input('var_188:',min_value=0.1, max_value=100.0, value=1.0)
var_189 = st.number_input('var_189:',min_value=0.1, max_value=100.0, value=1.0)
var_190 = st.number_input('var_190:',min_value=0.1, max_value=100.0, value=1.0)
var_191 = st.number_input('var_191:',min_value=0.1, max_value=100.0, value=1.0)
var_192 = st.number_input('var_192:',min_value=0.1, max_value=100.0, value=1.0)
var_193 = st.number_input('var_193:',min_value=0.1, max_value=100.0, value=1.0)
var_194 = st.number_input('var_194:',min_value=0.1, max_value=100.0, value=1.0)
var_195 = st.number_input('var_195:',min_value=0.1, max_value=100.0, value=1.0)
var_196 = st.number_input('var_196:',min_value=0.1, max_value=100.0, value=1.0)
var_197 = st.number_input('var_197:',min_value=0.1, max_value=100.0, value=1.0)
var_198 = st.number_input('var_198:',min_value=0.1, max_value=100.0, value=1.0)
var_199 = st.number_input('var_199:',min_value=0.1, max_value=100.0, value=1.0)





if st.button('Predict Price'):
    target = predict(var_0,
var_1,
var_2,
var_3,
var_4,
var_5,
var_6,
var_7,
var_8,
var_9,
var_10,
var_11,
var_12,
var_13,
var_14,
var_15,
var_16,
var_17,
var_18,
var_19,
var_20,
var_21,
var_22,
var_23,
var_24,
var_25,
var_26,
var_27,
var_28,
var_29,
var_30,
var_31,
var_32,
var_33,
var_34,
var_35,
var_36,
var_37,
var_38,
var_39,
var_40,
var_41,
var_42,
var_43,
var_44,
var_45,
var_46,
var_47,
var_48,
var_49,
var_50,
var_51,
var_52,
var_53,
var_54,
var_55,
var_56,
var_57,
var_58,
var_59,
var_60,
var_61,
var_62,
var_63,
var_64,
var_65,
var_66,
var_67,
var_68,
var_69,
var_70,
var_71,
var_72,
var_73,
var_74,
var_75,
var_76,
var_77,
var_78,
var_79,
var_80,
var_81,
var_82,
var_83,
var_84,
var_85,
var_86,
var_87,
var_88,
var_89,
var_90,
var_91,
var_92,
var_93,
var_94,
var_95,
var_96,
var_97,
var_98,
var_99,
var_100,
var_101,
var_102,
var_103,
var_104,
var_105,
var_106,
var_107,
var_108,
var_109,
var_110,
var_111,
var_112,
var_113,
var_114,
var_115,
var_116,
var_117,
var_118,
var_119,
var_120,
var_121,
var_122,
var_123,
var_124,
var_125,
var_126,
var_127,
var_128,
var_129,
var_130,
var_131,
var_132,
var_133,
var_134,
var_135,
var_136,
var_137,
var_138,
var_139,
var_140,
var_141,
var_142,
var_143,
var_144,
var_145,
var_146,
var_147,
var_148,
var_149,
var_150,
var_151,
var_152,
var_153,
var_154,
var_155,
var_156,
var_157,
var_158,
var_159,
var_160,
var_161,
var_162,
var_163,
var_164,
var_165,
var_166,
var_167,
var_168,
var_169,
var_170,
var_171,
var_172,
var_173,
var_174,
var_175,
var_176,
var_177,
var_178,
var_179,
var_180,
var_181,
var_182,
var_183,
var_184,
var_185,
var_186,
var_187,
var_188,
var_189,
var_190,
var_191,
var_192,
var_193,
var_194,
var_195,
var_196,
var_197,
var_198,
var_199)
    st.success(f'The predicted price of the diamond is ${target[0]:.2f} USD')
