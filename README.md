### Prototype data models for understanding and predicting weather-related incidents on Great Britain's rail network

This repo presents prototype data models for predicting weather-related incidents on GB's rail network. Two case study examples are demonstrated in the context of delays caused primarily by high winds and temperatures, respectively.

### Brief background

The model was initially developed in a research project which was intended to establish a modelling framework for exploration of the applicability of existing data resources to specific industry problems. The methodology was demonstrated in the context of wind-related 8 delays on the rail network in Great Britain. This case study examined the sub-networks most susceptible to delays as a result of high-wind events; and it identified appropriate data sources that could support the decision-making process around placement of trackside weather stations to monitor the impact of the high wind speeds. 

The study investigates a variety of relevant data sources, including historical records of wind-related incidents, and the weather and vegetation conditions around the incident locations. It shows on how those data could be exploited and modelled with logistic regression analysis. A number of factors contributing to wind-related delay incidents are identified and used for making predictions of likelihood of such incidents occurrences. 

The prototype data model is generic and should also be adaptable to other contexts, e.g. with other categories of weather-related incidents or a wider range of common industry tasks.

---


##### *Journal article*

- **Fu, Q**. and [Easton, J. M.](https://www.birmingham.ac.uk/staff/profiles/eese/easton-john.aspx) (2018), Prediction of weather-related incidents on the rail network: prototype data model for wind-related delays in Great Britain. ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering 4(3): 04018027. [doi:10.1061/AJRUA6.0000975](https://doi.org/10.1061/AJRUA6.0000975). 

##### *Technical report*

- **Fu, Q**. and [Easton, J. M.](https://www.birmingham.ac.uk/staff/profiles/eese/easton-john.aspx) (2018), Establishing the value of existing data sources as supporting datasets for improved decision-making. Approved by Network Rail. (*Unpublished*) 

##### *Seminar*

- **Fu, Q.** Understanding and predicting weather-related incidents on the rail network: case studies of wind-/heat-related incidents. Invited seminar at the Institute for Risk and Uncertainty, University of Liverpool, Liverpool, UK, 8 January 2019. 


##### *Conference papers*

- ***(Forthcoming)*** [Jaroszweski, D. J.](https://www.birmingham.ac.uk/schools/engineering/civil-engineering/people/profile.aspx?ReferenceId=30587&Name=dr-david-jaroszweski), **Fu, Q.** and [Easton, J. M.](https://www.birmingham.ac.uk/staff/profiles/eese/easton-john.aspx) (2018), A data model for heat-related rail buckling: implications for operations, maintenance. Accepted for oral presentation at the upcoming 12th World Congress on Railway Research, Tokyo, Japan, 28 October - 1 November 2019. 
- **Fu, Q.**, [Jaroszweski, D. J.](https://www.birmingham.ac.uk/schools/engineering/civil-engineering/people/profile.aspx?ReferenceId=30587&Name=dr-david-jaroszweski) and [Easton, J. M.](https://www.birmingham.ac.uk/staff/profiles/eese/easton-john.aspx) (2018), A prototype model for understanding heat-related rail incidents: a case study on the Anglia area in Great Britain, paper presented at the [8th International Conference on Railway Engineering, IET London: Savoy Place, 16-17 May 2018](https://digital-library.theiet.org/content/conferences/cp742) \[[Presentation](https://tv.theiet.org/?videoid=12228)\] \[[Q&A](https://tv.theiet.org/?videoid=12230)\] (*Access Code for the presentation and Q&A session can be provided upon request*). Published in the Proceedings of the 8th International Conference on Railway Engineering, Institution of Engineering and Technology. [doi:10.1049/cp.2018.0071](http://digital-library.theiet.org/content/conferences/10.1049/cp.2018.0071). 
- **Fu, Q**. and [Easton, J. M.](https://www.birmingham.ac.uk/staff/profiles/eese/easton-john.aspx) (2018), A data model for prediction of weather-related rail incidents: a case-study example of wind-related incidents on the rail network in Great Britain, paper presented at the 50th Annual Conference of the Universities’ Transport Study Group, London, UK, 3-5 January 2018. (*Unpublished*.) 
- **Fu, Q**. and [Easton, J. M.](https://www.birmingham.ac.uk/staff/profiles/eese/easton-john.aspx) (2016), How does existing data improve decision making? A case study of wind-related incidents on rail network in Great Britain, paper presented at the 
  [2016 International Conference on Railway Engineering, Brussels, Belgium, 12-13 May 2016](
  https://digital-library.theiet.org/content/conferences/cp703) \[[Presentation](https://tv.theiet.org/?videoid=8607)\]. Published in the Proceedings of the International Conference on Railway Engineering (ICRE 2016), Institution of Engineering and Technology. 
  [doi:10.1049/cp.2016.0515](https://ieeexplore.ieee.org/document/7816543/). 


##### *Python packages (open for further development)*

- **pydriosm** [[source code](https://github.com/mikeqfu/pydriosm)] - Download, read/parse and import/export OSM data extracts. 
- **pyrcs** [[source code](https://github.com/mikeqfu/pyrcs)] - A small tool for collecting railway codes used in GB. 
- **pyhelpers** [[source code](https://github.com/mikeqfu/pyhelpers)] - A small toolkit of some helper functions to facilitate writing Python code.
