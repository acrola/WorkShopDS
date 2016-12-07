indicators_dict = {
    'EG.ELC.ACCS.ZS': 'Access to electricity (% of population)', \
    'EG.USE.ELEC.KH.PC': 'Electric power consumption (kWh per capita)', \
    'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)', \
    'TX.VAL.FUEL.ZS.UN': 'Fuel exports (% of merchandise exports)', \
    'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)', \
    'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)', \
    'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)', \
    'SL.TLF.CACT.FE.ZS': 'Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)', \
    'SL.TLF.CACT.MA.ZS': 'Labor force participation rate, male (% of male population ages 15+) (modeled ILO estimate)', \
    'SL.TLF.CACT.ZS': 'Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)', \
    'SP.DYN.LE00.FE.IN': 'Life expectancy at birth, female (years)', \
    'SP.DYN.LE00.MA.IN': 'Life expectancy at birth, male (years)', \
    'EN.POP.DNST': 'Population density (people per sq. km of land area)', \
    'SP.POP.GROW': 'Population growth (annual %)', \
    'IT.MLT.MAIN.P2': 'Telephone lines (per 100 people)', \
    'SL.UEM.TOTL.FE.ZS': 'Uennemploymt, female (% of female labor force) (modeled ILO estimate)', \
    'SL.UEM.TOTL.MA.ZS': 'Unemployment, male (% of male labor force) (modeled ILO estimate)', \
    'SL.UEM.TOTL.ZS': 'Unemployment, total (% of total labor force) (modeled ILO estimate)', \
    'SL.UEM.1524.FE.ZS': 'Unemployment, youth female (% of female labor force ages 15-24) (modeled ILO estimate)', \
    'SL.UEM.1524.MA.ZS': 'Unemployment, youth male (% of male labor force ages 15-24) (modeled ILO estimate)', \
    'SL.UEM.1524.ZS': 'Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)', \
    'IT.CEL.SETS.P2': 'Mobile cellular subscriptions (per 100 people)',\
'IT.TEL.TOTL.P2': 'Fixed line and mobile cellular subscriptions (per 100 people)',\
'SH.STA.ACSN': 'Improved sanitation facilities (% of population with access)',\
'SH.H2O.SAFE.ZS': 'Improved water source (% of population with access)',\
'EG.ELC.HOUS.ZS': 'Household electrification rate (% of households)',\
'SP.DYN.TFRT.IN': 'Fertility rate, total (births per woman)',\
'SP.ADO.TFRT': 'Adolescent fertility rate (births per 1,000 women ages 15-19)',\
'EG.ELC.ACCS.ZS': 'Access to electricity (% of population)',\
'IS.ROD.ALLS.ZS': 'Access to an all-season road (% of rural population)',\
'SP.DYN.LE00.IN': 'Life expectancy at birth, total (years)',\
'IC.REG.DURS         ': 'Time required to start a business (days)',\
'SE.PRM.UNER.ZS': 'Children out of school (% of primary school age)',\
'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)',\
'TX.VAL.FUEL.ZS.UN': 'Fuel exports (% of merchandise exports)',\
'TX.VAL.TECH.MF.ZS': 'High-technology exports (% of manufactured exports)',\
'NE.EXP.GNFS.KD.ZG': 'Exports of goods and services (annual % growth)',\
'NY.GDP.PCAP.PP.KD': 'GDP per capita, PPP (constant 2011 international $)',\
'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',\
'NY.GDP.PCAP.KD': 'GDP per capita (constant 2005 US$)',\
'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)',\
'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)',\
'UIS.NERA.2': 'Adjusted net enrolment rate, lower secondary, both sexes (%)',\
'UIS.NERA.2.GPI': 'Adjusted net enrolment rate, lower secondary, gender parity index (GPI)',\
'SE.PRM.TENR': 'Adjusted net enrolment rate, primary, both sexes (%)',\
'UIS.NERA.1.GPI': 'Adjusted net enrolment rate, primary, gender parity index (GPI)',\
'UIS.NERA.3': 'Adjusted net enrolment rate, upper secondary, both sexes (%)',\
'UIS.NERA.3.GPI': 'Adjusted net enrolment rate, upper secondary, gender parity index (GPI)',\
'SE.ADT.LITR.ZS': 'Adult literacy rate, population 15+ years, both sexes (%)',\
'UIS.LR.AG15T99.GPI': 'Adult literacy rate, population 15+ years, gender parity index (GPI)',\
'OECD.TSAL.1.E10': 'Annual statutory teacher salaries in public institutions in USD. Primary. 10 years of experience',\
'OECD.TSAL.2.E10': 'Annual statutory teacher salaries in public institutions in USD. Lower Secondary. 10 years of experience',\
'OECD.TSAL.2.ETOP': 'Annual statutory teacher salaries in public institutions in USD. Lower Secondary. Top of scale',\
'OECD.TSAL.1.ETOP': 'Annual statutory teacher salaries in public institutions in USD. Primary. Top of scale',\
'OECD.TSAL.3.E15': 'Annual statutory teacher salaries in public institutions in USD. Upper Secondary. 15 years of experience',\
'OECD.TSAL.3.ETOP': 'Annual statutory teacher salaries in public institutions in USD. Upper Secondary. Top of scale',\
'SE.COM.DURS': 'Duration of compulsory education (years)',\
'UIS.LR.AG65': 'Elderly literacy rate, population 65+ years, both sexes (%)',\
'SE.TOT.ENRR': 'Gross enrolment ratio, primary to tertiary, both sexes (%)',\
'IT.NET.USER.P2': 'Internet users (per 100 people)',\
'SL.TLF.PRIM.ZS': 'Labor force with primary education (% of total)',\
'UIS.ROFST.2': 'Rate of out-of-school adolescents of lower secondary school age, both sexes (%)',\
'SE.SEC.ENRL.TC.ZS': 'Pupil-teacher ratio in secondary education (headcount basis)',\
'SE.XPD.TOTL.GD.ZS': 'Government expenditure on education as % of GDP (%)',\
'UIS.XUNIT.US.3.FSGOV': 'Government expenditure per upper secondary student (US$)'\


}
