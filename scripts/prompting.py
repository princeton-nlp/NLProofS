import re
import json
import openai
import argparse
from tqdm import tqdm
from time import sleep


prompt_base = {
    1: """Hypothesis: if a fossil of a bird cannot be identified then that kind of bird is probably extinct
Context:
sent1: identifying is similar to determining 
sent2: if a fossil is of an organism that cannot be identified then that organism is probably extinct 
sent3: discovering something usually requires seeing that something 
sent4: a dinosaur is a kind of extinct animal 
sent5: fossils can be used as evidence for the ancient environment 
sent6: dead means not alive 
sent7: fossil means preserved remains of ancient organisms 
sent8: a type is synonymous with a kind 
sent9: nonliving means not living 
sent10: if a living thing dies then that living thing is dead 
sent11: a description sometimes provides information 
sent12: to discover means to find 
sent13: a bird is a kind of animal 
sent14: cannot means not be able to 
sent15: an animal is a member of an animal species 
sent16: remains mean parts of a dead organism 
sent17: endangered means low in population 
sent18: if there is none of a thing then that thing does not exist 
sent19: preserved means from the past / from long ago 
sent20: to identify means to discover 
sent21: existence is similar to living 
sent22: identifying sometimes requires examining 
sent23: if the population of an animal decreases then that animal may no longer be found in that place 
sent24: an animal is a kind of organism 
sent25: fossils can be used to study the history of organisms and environments on earth
Proof: sent13 & sent24 -> int1: a bird is a kind of organism; int1 & sent2 -> hypothesis;

Hypothesis: an animal requires water and air and food for survival
Context:
sent1: breathing in is when animals inhale air into their lungs 
sent2: animals / living things require water for survival 
sent3: requiring something means needing that something 
sent4: if the amount of available food and water decreases in an environment then animals may leave that environment to find food and water 
sent5: to depend on / to rely on / to need means to require 
sent6: survive means live 
sent7: to eat means to consume food 
sent8: an animal / bacterium requires oxygen for survival / to breathe 
sent9: a lack of something that a living thing requires prevents the survival of that living thing 
sent10: breathing is when a lung converts from oxygen in air into oxygen in blood 
sent11: lack of food causes starvation 
sent12: oxygen can be found in air 
sent13: an animal / living thing requires nutrients for survival 
sent14: to be used for something means to be required by that something 
sent15: survival means to survive 
sent16: if something required by an organism is depleted then that organism must replenish that something 
sent17: cold environments usually have little food for animals 
sent18: requiring is similar to needing help 
sent19: an animal requires warmth for survival 
sent20: needing something means depending on that something 
sent21: an animal needs to eat food for nutrients 
sent22: survive means live 
sent23: to breathe in means to absorb air 
sent24: food is what an animal eats 
sent25: the amount of something is similar to the availability of something
Proof: sent12 & sent8 -> int1: an animal requires air for survival; int1 & sent2 -> int2: an animal requires water and air for suvival; sent13 & sent21 -> int3: an animal requires food for survival; int2 & int3 -> hypothesis;

Hypothesis: stars that are blue in color are hottest in temperature
Context:
sent1: a hot substance is a source of heat 
sent2: the surface of the sun is extremely hot in temperature with values as high as 20 000 000  c 
sent3: warm is similar to hot 
sent4: a type of something is a kind of property of that something 
sent5: blue stars are the hottest in temperature 
sent6: contains means located in 
sent7: made up of means contains / made of 
sent8: characteristic means property 
sent9: heat energy is synonymous with thermal energy 
sent10: the sun is a source of radiation / heat called sunlight 
sent11: the properties of something can be used to identify / used to describe that something 
sent12: classifying means grouping objects / materials by their properties 
sent13: measuring sometimes requires recording / learning an amount 
sent14: the properties of something are used for describing that something 
sent15: heating means adding heat 
sent16: a type is synonymous with a kind 
sent17: temperature is a measure of kinetic energy of molecules 
sent18: measuring is a kind of observing 
sent19: blue is a kind of color 
sent20: temperature is a property of air mass and includes ordered values of cold / warm 
sent21: a surface is a part of a material 
sent22: heat means heat energy 
sent23: both means two 
sent24: heat is a kind of energy 
sent25: surface type is a kind of characteristic
Proof: sent19 & sent5 -> hypothesis;

Hypothesis: as mileage per gallon of oil increases, the amount of time that oil is available will be extended
Context:
sent1: performing a task in less time / more quickly / faster has a positive impact on a person 's life 
sent2: a measure of time is a length of time 
sent3: to increase something can mean to extend somthing 
sent4: using resources decreases those resources 
sent5: as a system / device grows old , the amount of energy will used by the system / device will increase 
sent6: gasoline is a kind of fuel 
sent7: to last longer means to be available longer 
sent8: oil is a source of gasoline 
sent9: how long something takes is a kind of measurement of time 
sent10: making something available is similar to providing 
sent11: fuel efficient means uses less fuel 
sent12: as the use of a resource decreases , the length of time that resource being available will increases 
sent13: as the use of something decreases , the use of the source of that something will decrease 
sent14: fuel is a kind of resource 
sent15: to be used for something means to be required by that something 
sent16: using less of a resource increases the availability of that resource 
sent17: a supply of something is a source of that something 
sent18: increase means more 
sent19: to reduce means to decrease 
sent20: the amount of something is similar to the availability of something 
sent21: as mileage per gallon of gasoline increases , the amount of gasoline used will decrease 
sent22: to add means to increase 
sent23: fuel supply is a kind of resource 
sent24: if a resource is limited in supply then that resource will run out 
sent25: to provide means to supply
Proof: sent14 & sent6 -> int1: gasoline is a kind of resource; int1 & sent12 -> int2: as the use of gasoline decreases, the length of time that gasoline is available will increase; int2 & sent21 -> int3: as mileage per gallon of gasoline increases, the length of time that gasoline is available will increase; int3 & sent8 -> int4: as mileage per gallon of oil increases, the amount of time that oil is available will increase; int4 & sent3 -> hypothesis;

Hypothesis: the firecracker stores chemical energy as its original energy
Context:
sent1: if something emits something else then that something increases the amount of that something else 
sent2: phase means state 
sent3: fire transfers heat through waves 
sent4: heat means heat energy 
sent5: a type of something is a kind of property of that something 
sent6: combustion is a kind of chemical change 
sent7: temperature / heat energy is a property of objects / weather and includes ordered values of cold / cool / warm / hot 
sent8: amount is a property of something and includes ordered values of none / least / little / some / half / much / many / most / all 
sent9: to be a source of something means to cause that something 
sent10: if an object / a substance makes something then that object / that substance is a source of that thing 
sent11: a firecracker converts chemical energy into sound energy and light energy and heat energy 
sent12: sound means sound energy 
sent13: fire is a kind of chemical reaction 
sent14: a source of something increases the amount of that something 
sent15: if something releases something else then that something is the source of that something else 
sent16: different substances usually have different amount of chemical energy 
sent17: where something comes from is a source of that something 
sent18: heat energy can change the state of matter 
sent19: energy transformation means one kind of energy changes into another kind of energy 
sent20: light means light energy 
sent21: if something converts one kind of energy into other kinds of energy then that something originally stores that energy as the first kind of energy 
sent22: heating means adding heat 
sent23: chemicals means substances 
sent24: if something is a source of something else then that something usually contains that something else 
sent25: heat energy is synonymous with thermal energy
Proof: sent11 & sent21 -> hypothesis;

Hypothesis: humans throwing garbage into a stream causes harm to the stream
Context:
sent1: absorbing something harmful has a negative impact on a thing 
sent2: objects in an environment are a part of that environment 
sent3: discarding is similar to throwing 
sent4: a pond is a kind of body of water 
sent5: landfills have a negative impact on the environment / communities 
sent6: an landfill is a source of pollution 
sent7: a stream is a kind of moving body of water 
sent8: humans discarding waste in an environment causes harm to that environment 
sent9: water pollution is when humans pollute the environment with pollutants 
sent10: a puddle is a kind of body of water 
sent11: a drop / droplet is a kind of liquid of small amount 
sent12: garbage means waste 
sent13: an environment means an area 
sent14: a body of water is a kind of environment 
sent15: debris can cause harm to an organism 
sent16: trash is synonymous with waste 
sent17: if something releases something else then that something is the source of that something else 
sent18: pollutants have a negative impact on the environment / air quality 
sent19: humans move waste to a landfill for disposal / storage 
sent20: removing waste is a kind of function 
sent21: if something has a negative impact on something else then increasing the amount of that something has a negative impact on that something else 
sent22: the effects of something is a property of that something 
sent23: polluting means something harmful / something poisonous is added in an environment causes harm in the environment 
sent24: pollution is a source of pollutants 
sent25: waste must be removed
Proof: sent12 & sent3 & sent8 -> int1: humans throwing garbage in an environment causes harm to that environment; sent14 & sent7 -> int2: a stream is a kind of environment; int1 & int2 -> hypothesis;

Hypothesis: the sun will appear larger than other stars because it is the closest star to earth
Context:
sent1: to move away means to increase distance 
sent2: size is a property of objects and includes ordered values of microscopic / tiny / small / medium / large 
sent3: as distance from an object decreases , that object will appear larger 
sent4: as distance from an object decreases , the pull of gravity on that object increases 
sent5: the sun is a kind of yellow dwarf 
sent6: stars are located light years apart from each other 
sent7: proximity means distance 
sent8: near / close is the opposite of far / away 
sent9: the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year 
sent10: close means low in distance 
sent11: as the size of a light source increases , that light will appear brighter 
sent12: height is a property of the size of an object 
sent13: moving away from the source increases the distance 
sent14: being in the sun is synonymous with being in the sunlight 
sent15: the sun is the star that is closest to earth 
sent16: the sun is average in size for a star in our galaxy 
sent17: distance moved / distance travelled is a measure of how far an object moves 
sent18: as the distance from an object increases , the force of gravity on that object will decrease 
sent19: a star is a kind of celestial object / celestial body 
sent20: measuring sometimes requires recording / learning an amount 
sent21: a dwarf planet usually is much smaller in size / in mass than other planets 
sent22: earth is a kind of planet 
sent23: a planet is a kind of celestial object / body 
sent24: planets orbit stars 
sent25: distance is a property of space and includes ordered values of close / far
Proof: sent22 & sent23 -> int1: earth is a kind of celestial object; int1 & sent19 & sent3 -> int2: as the distance from a star to earth decreases, the star will appear larger; int2 & sent15 -> hypothesis;""",
    2: """Hypothesis: the water in the bucket will evaporate
Context:
sent1: sunlight shining means sunlight is provided 
sent2: if something is in the sunlight then that something will absorb solar energy 
sent3: to be in the sun means to be in the sunlight 
sent4: if a substance absorbs solar energy then that substance will increase in temperature 
sent5: cooling means temperature decreases 
sent6: evaporation means a substance changes from a liquid into a gas by increasing heat energy 
sent7: thermal energy is a kind of energy 
sent8: heat is a kind of energy 
sent9: heat means the transfer of thermal energy 
sent10: water absorbs light energy 
sent11: if heat is absorbed from a source then that heat source will cool 
sent12: intensity of sunlight is similar to amount of sunlight 
sent13: heat energy is synonymous with thermal energy 
sent14: warm up means increase temperature 
sent15: if heat is added to a substance then that substance absorbs that heat 
sent16: absorbing energy causes objects / materials / substances to heat 
sent17: water is a kind of substance 
sent18: as temperature during the day increases , the temperature in an environment will increase 
sent19: as the temperature of a liquid increases , the rate of evaporation of that liquid will increase 
sent20: solar energy can warm up the air 
sent21: as the sunlight absorbed by the object increases , the temperature of the object will increase more 
sent22: being in the sun is synonymous with being in the sunlight 
sent23: if a liquid disappears then that liquid probably evaporated 
sent24: a bucket of water is in the sunlight 
sent25: as the amount of water in a body of water increases , the amount of water evaporated from the body of water will increase
Proof: sent2 & sent24 -> int1: the water in the bucket will absorb solar energy; int1 & sent17 & sent4 -> int2: the water in the bucket will increase in temperature; int2 & sent17 & sent6 -> hypothesis;

Hypothesis: a mutation in sperm or egg of a parent can cause a new trait to appear in the parent 's offspring
Context:
sent1: dna are a vehicle for passing genes from parent to offspring 
sent2: sexual reproduction is a source of genetic variation / naturally occurring variation in offspring / in a species 
sent3: reproduction increases the number / population of a living thing 
sent4: genetic information contains instructions for the passage of traits from one generation to the next 
sent5: if an organism passes on its traits then future generations will have those traits 
sent6: producing is similar to causing 
sent7: sexual reproduction increases genetic diversity 
sent8: some animals have offspring by laying eggs 
sent9: genes contains genetic information 
sent10: fertilization is when an egg cell becomes fertilized 
sent11: one parent is the source of 50% of the genes in a fertilized egg through sexual reproduction 
sent12: genes is a vehicle for passing genetic information from one generation to the next 
sent13: each sex cell provides half the number of chromosomes in a fertilized egg through sexual reproduction 
sent14: to produce means to result in 
sent15: information in an organism 's chromosomes cause genetic traits to be passed down to that organism 's offspring 
sent16: a sperm cell is a kind of sex cell 
sent17: a mutation in the sex cells of a parent can cause a new trait to appear in the parent 's offspring 
sent18: an egg cell is a kind of sex cell 
sent19: to reproduce means to have / to produce offspring 
sent20: reproduction is when an organism passes genetic information from itself to its offspring 
sent21: trait is a kind of genetic information 
sent22: genetic / hereditary means of genes / heredity 
sent23: trait means property 
sent24: genotype of an organism means the genes of an organism 
sent25: reproductive behavior is an inherited characteristic
Proof: sent16 & sent17 & sent18 -> hypothesis;

Hypothesis: a lack of moisture prevents the survival of plants in the desert
Context:
sent1: requiring too much of a resource has a negative impact on the availiability of that resource 
sent2: plants require water for survival 
sent3: a plant is a kind of living thing 
sent4: moisture is a form of water 
sent5: disrupting something from reaching something else decreases the availability of that something 
sent6: if a living thing is destroyed then the resources used by that living thing will become available 
sent7: when available resources decrease in an environment , organisms have to conserve those resources 
sent8: if something has a negative impact on the survival of an organism then that organism may be unable to survive 
sent9: a plant requires soil for survival / to grow 
sent10: as available water decreases , the population of plants will decrease 
sent11: a plant requires sunlight for photosynthesis 
sent12: as a population of organisms increases , the resources used by those organisms will decrease 
sent13: if something required for a process is not produced then that process is prevented from occurring / cannot occur 
sent14: a desert environment is low in availability of water 
sent15: as a resource required by an organism decreases , the population of that organisms will decrease 
sent16: a cactus lives in the desert 
sent17: if the amount of available food and water decreases in an environment then animals may leave that environment to find food and water 
sent18: a lack of something that a living thing requires prevents the survival of that living thing 
sent19: if water vapor is limited to reach a location , then that location will be dry 
sent20: if a resource is not replaced then the resource has low availability 
sent21: a plant requires a habitat for survival 
sent22: a tree requires sunlight to grow 
sent23: a plant requires a specific climate to grow and survive 
sent24: a plant requires water to grow 
sent25: a plant requires sunlight to grow
Proof: sent18 & sent3 -> int1: a lack of something that a plant requires prevents the survival of that plant; int1 & sent2 -> int2: a lack of water prevents the survival of plants; int2 & sent14 -> int3: a lack of water prevents the survival of plants in the desert; int3 & sent4 -> hypothesis;

Hypothesis: cold fronts can cause thudnerstorms of rain as they pass by
Context:
sent1: temperature is a property of air mass and includes ordered values of cold / warm 
sent2: if a jet stream moves south of a location then that location will experience cold weather 
sent3: when clouds become cold , those clouds will condense 
sent4: as the amount of water in air condensing into clouds decreases , the amount of precipitation will decrease 
sent5: a hurricane has large amounts of rain 
sent6: lack of moisture in the air causes low amounts of rainfall 
sent7: as air pressure decreases , the chance of rain will increase 
sent8: rain is a kind of precipitation 
sent9: cold fronts cause thunderstorms as they pass by 
sent10: clouds are usually found in the sky 
sent11: rainfall means precipitation 
sent12: rainfall means the amount of precipitation 
sent13: a thunderstorm is a kind of storm 
sent14: if weather is stormy then there is a greater chance of rain 
sent15: rainstorm means rain storm 
sent16: precipitation is when water falls from the sky 
sent17: a thunderstorm can cause a tornado 
sent18: types of clouds can be used to predict the possibility of tornadoes developing in a location 
sent19: if an event is required for a process then that event must occur before that process can occur 
sent20: water vapor condensing in clouds causes rain 
sent21: to cause an event means to make that event more likely 
sent22: a storm is a source of precipitation 
sent23: a storm is usually a source of strong winds 
sent24: more likely means increased likelihood 
sent25: an area receiving rain means rain falling in that area
Proof: sent22 & sent8 -> int1: a storm is a source of rain; int1 & sent13 -> int2: a thunderstorm is a source of rain; int2 & sent9 -> hypothesis;

Hypothesis: the earth rotating on its tilted axis causes the cycles of day and night on earth
Context:
sent1: a phase change is a kind of physical change 
sent2: motion / movement means moving / to move 
sent3: afternoon is a part of day time 
sent4: hours are a kind of unit for measuring time 
sent5: the afternoon is a part of the day 
sent6: due to means caused by 
sent7: the earth rotates on its tilted axis 
sent8: the sunlight occurs during the day 
sent9: phase means state 
sent10: can be means able to be 
sent11: an example of a physical change is a phase change 
sent12: earth is a kind of planet 
sent13: all the time means at day and at night 
sent14: if something used to be in the past then that something has changed 
sent15: as the rotation speed of a planet increases , the length of day and night will decrease on that planet 
sent16: to change means to cause a change 
sent17: rotating is similar to moving 
sent18: days ( d ) are a metric unit used for measuring time generally used for values between 1 and 365 
sent19: rotate means turn 
sent20: a phase change is when matter / a substance changes from one state of matter into another state of matter 
sent21: a planet rotating causes cycles of day and night on that planet 
sent22: approximately means about 
sent23: rotation is the circular movement of an object around a center / axis 
sent24: to convert means to change 
sent25: to rotate means to complete a rotation
Proof: sent12 & sent7 -> int1: the earth is a planet that rotates on its tilted axis; int1 & sent21 -> hypothesis;

Hypothesis: a compound can only be broken down in chemical change but not physcial changes
Context:
sent1: chemical reaction means chemical bonds change 
sent2: breaking down an object changes that object 's shape / that object 's mass 
sent3: a compound being broken down is a kind of chemical change 
sent4: except means not 
sent5: a compound can be chemically separated into the elements that it is made of 
sent6: amount is a property of something and includes ordered values of none / least / little / some / half / much / many / most / all 
sent7: if two substances together form a mixture then those substances can be separated from one another by physical changes 
sent8: an element cannot be decomposed into two or more different substances by simple chemical methods 
sent9: chemical reactions do not cause the total number of atoms to change 
sent10: an example of a physical change is breaking an object 
sent11: break apart means break down 
sent12: changed is the opposite of unchanged 
sent13: separating is similar to breaking 
sent14: breaking an object into pieces changes the shape of the object 
sent15: break down is the opposite of put together 
sent16: chemical bonds are formed by chemically combining elements / atoms / molecules 
sent17: a chemical property is the opposite of a physical property 
sent18: when two substances together form a compound then those substances cannot be physically separated 
sent19: to break down means to break into smaller pieces 
sent20: matter / energy is always conserved in any physical or chemical process 
sent21: break down means change from a whole into pieces 
sent22: a physical property is a kind of property 
sent23: a chemical property is a kind of property 
sent24: stay the same means not changing 
sent25: if something changes , then that something is not the same
Proof: sent3 & sent5 -> int1: a compound can only be separated into the elements by chemical change; int1 & sent18 -> hypothesis;

Hypothesis: the 150 ml water has a freezing point of 0 c
Context:
sent1: liquid has a a lower melting point than solid 
sent2: melting is when solids are heated above their melting point 
sent3: melting is a kind of phase change 
sent4: the freezing point of water is 0 c 
sent5: state of matter is a property of matter and includes ordered values of solid / liquid / gas 
sent6: ice is a kind of solid 
sent7: freezing point is similar to melting point 
sent8: milliliters ml is a metric unit used for measuring volume generally used for values between 1 and 1000 
sent9: as heat is transferred from something to something else , the temperature of that something will decrease 
sent10: freezing causes a solid to form 
sent11: volume is similar to amount 
sent12: freezing means cold temperatures 
sent13: matter in the solid phase has definite volume 
sent14: freezing point is a property of a substance / material 
sent15: cooling means temperature decreases 
sent16: melting point means temperature at which a solid melts / above which a solid melts 
sent17: ice is colder in temperature than water 
sent18: melting point is a property of a substance / material 
sent19: a phase change is when matter / a substance changes from one state of matter into another state of matter 
sent20: volume has no impact on boiling / melting / freezing point 
sent21: there is a 100 ml water 
sent22: cold means low in temperature 
sent23: if an object is placed in a substance / in a location that is colder then that object will cool to the same temperature of that substance / of that location 
sent24: mass has no impact on boiling / melting / freezing point 
sent25: there is a 150 ml water
Proof: sent21 & sent25 & sent8 -> int1: the 100 ml water and the 150 ml water have different volumes; int1 & sent20 -> int2: different volumes have no impact on the freezing point of the 100ml water and 150 ml water; int2 & sent4 -> hypothesis;""",
    3: """Hypothesis: the offspring of two tay-sachs carriers will have 25% probability to have the tay-sachs disease
Context:
sent1: if only recessive genes are present , then the recessive trait will be visible / expressed 
sent2: determine is similar to cause 
sent3: disease means illness 
sent4: inherited characteristics are a kind of hereditary information 
sent5: blood type is an inherited characteristic 
sent6: amount is a property of something and includes ordered values of none / least / little / some / half / much / many / most / all 
sent7: the inherited characteristics of the parents can be used to predict inherited characteristics of the offspring 
sent8: parents produce offspring 
sent9: inheriting is when an inherited characteristic is copied / is passed from parent to offspring by genetics / dna 
sent10: if both dominant genes are present , then the dominant trait will be visible / expressed 
sent11: genes contains genetic information 
sent12: likelihood is similar to probability 
sent13: offspring receives half of the genes from each parent 
sent14: genes are able to determine the inherited characteristics of a living thing 
sent15: dna are a vehicle for passing genes from parent to offspring 
sent16: a gene is a kind of unit of hereditary information 
sent17: genes are located on chromosomes 
sent18: a heterozygous organism contains one dominant gene and one recessive gene 
sent19: genes is a vehicle for passing genetic information from one generation to the next 
sent20: traits can be determined by one pair / many pairs of genes in an organism 
sent21: genotype of an organism means the genes of an organism 
sent22: a tay-sachs disease carrier have one dominant and one recessive tay-sachs gene 
sent23: crossing two heterozygous dominant organisms causes their offspring to be homozygous recessive of 25% probability 
sent24: 25% probability is similar to one in four 
sent25: a homozygous recessive organism contains only recessive genes
Proof: sent18 & sent22 -> int1: a tay-sachs disease carrier is heterozygous; int1 & sent23 -> int2: the offspring of two tay-sachs carriers will have 25% probability to be homozygous recessive in tay-sachs disease; int2 & sent25 -> int3: the offspring of two tay-sachs carriers will have 25% probability to contain only recessive genes for tay-sachs disease; int3 & sent1 -> hypothesis;

Hypothesis: sulfur is a kind of element
Context:
sent1: sulfur is yellow in color 
sent2: chemical composition is a kind of property 
sent3: to tell the difference between things means to classify those things 
sent4: charge is a property of an object / a material / a substance and includes ordered values of negatively-charged / neutral / positively-charged 
sent5: including means containing 
sent6: a chemical property is a kind of property 
sent7: cannot means not be able to 
sent8: an element is identified by its number of protons 
sent9: chemical compounds contain chemical bonds between atoms / between elements 
sent10: to be formed by is to be the result of 
sent11: contains means located in 
sent12: if something is a part of something else then that something else contains that something 
sent13: amount is a property of something and includes ordered values of none / least / little / some / half / much / many / most / all 
sent14: chemical reactivity is a property of elements and includes ordered values of reactive / unreactive 
sent15: the properties of something can be used to identify / used to describe that something 
sent16: to classify means to decide what class something belongs to 
sent17: sulfur cannot be decomposed into different substances by simple chemical methods 
sent18: decomposing is similar to separating 
sent19: to decompose means to separate 
sent20: not is similar to the opposite of 
sent21: ability is a property of things and includes ordered values of able / unable / can / cannot 
sent22: both means two 
sent23: different is the opposite of the same 
sent24: able is the opposite of unable 
sent25: an element cannot be decomposed into two or more different substances by simple chemical methods
Proof: sent17 & sent25 -> hypothesis;

Hypothesis: the protist may be volvox
Context:
sent1: fertilization is a stage in the sexual reproduction process 
sent2: volvox reproduces sexually 
sent3: sexual reproduction increases genetic diversity 
sent4: producing is a kind of function 
sent5: reproduction increases the number / population of a living thing 
sent6: if something is given off of something , then something is the product of something 
sent7: a structure of something is synonymous with a part of that something 
sent8: the composition of something can be used to identify that something 
sent9: reptiles reproduce through internal fertilization 
sent10: volvox is a kind of protist 
sent11: to identify means to discover 
sent12: new cells with a full set of chromosomes are formed by fertilization 
sent13: the properties of something can be used to identify / used to describe that something 
sent14: fertilization is a kind of process 
sent15: a zygote is formed immediately after fertilization 
sent16: identifying is similar to determining 
sent17: a part of a living thing is a natural structure 
sent18: discovering something usually requires seeing that something 
sent19: if something is a part of something else then that something else contains that something 
sent20: a student observes the formation of zygotes by one of the protists 
sent21: a plant is a member of a plant species 
sent22: animals require fertilization to reproduce 
sent23: reproduction is when an organism passes genetic information from itself to its offspring 
sent24: reproduction ensures the continuation of a plant or animal species 
sent25: each sex cell provides half the number of chromosomes in a fertilized egg through sexual reproduction
Proof: sent1 & sent15 -> int1: if a zygote is formed then a sexual reproduction process has happened; int1 & sent20 -> int2: a sexual reproduction process has happened in the protists that the student observed; sent10 & sent2 -> int3: volvox is a kind of protist that reproduces sexually; int2 & int3 -> hypothesis;

Hypothesis: the light energy allows the student to see the specimen through the microscope
Context:
sent1: studying something usually requires seeing that something 
sent2: seeing requires light 
sent3: light enters the eye through the pupil 
sent4: a student can see the specimen through a microscope 
sent5: a retina is part of an eye for sensing light 
sent6: if an object reflects more light then that object is more easily seen 
sent7: if something is required for something else then that something allows that something else 
sent8: visible light is a kind of light 
sent9: if an object reflects light toward the eye then that object can be seen 
sent10: visible means able to be seen 
sent11: light reflecting off of an object causes that object to be visible to the observer 
sent12: viewing means observing light 
sent13: light is a kind of electromagnetic radiation 
sent14: waves can travel through matter 
sent15: observe means see 
sent16: visible light can be seen without using equipment 
sent17: brightness is a property of a light source and includes values of bright / dim 
sent18: if an object is transparent , then light will shine through that object without scattering 
sent19: solar energy is a kind of light 
sent20: when light enters the eye through the pupil , that light falls on the retina 
sent21: light rays means light 
sent22: light is a kind of energy 
sent23: sight means vision 
sent24: sunlight is a kind of light 
sent25: sight means to see
Proof: sent2 & sent22 -> int1: seeing requires light energy; int1 & sent7 -> int2: light energy allows things to be seen; int2 & sent4 -> hypothesis;

Hypothesis: gravity is the force that causes the marble to be pulled down on a planet
Context:
sent1: gravity accelerates an object while that object falls by pulling on it 
sent2: gravity is a kind of force 
sent3: exerting force on an object means a force acts on an object 
sent4: sinking is a kind of downward direction 
sent5: a force is a kind of push or pull on an object 
sent6: a drop / droplet is a kind of liquid of small amount 
sent7: the mass of an object causes the gravitational force exerted by that object 
sent8: a liquid is a kind of fluid 
sent9: gravity means gravitational pull / gravitational energy / gravitational force / gravitational attraction 
sent10: to cause means to result in 
sent11: colliding is a kind of touching 
sent12: particles in a liquid flows past each other 
sent13: force requires energy 
sent14: marble is a kind of object / material 
sent15: gravity causes objects that have mass / substances to be pulled down / to fall on a planet 
sent16: down is a kind of direction 
sent17: a droplet is a kind of little drop 
sent18: gravity pulls objects towards planets 
sent19: colliding means coming into a collision 
sent20: drop means decrease 
sent21: if something is dropped into a container of something else then that something is touching that something else 
sent22: when a force is applied to an object , that force is acting on that object 
sent23: flowing liquid can push objects 
sent24: empty into a body of water means enters a body of water 
sent25: to be lowered into a substance means to be placed into that substance
Proof: sent14 & sent15 & sent2 -> hypothesis;

Hypothesis: pesticides and fertilizers are source of pollution
Context:
sent1: harming an environment causes harm to the living things / to the organisms in that environment 
sent2: humans changing ecosystems / environments usually has a negative impact on an ecosystem / organisms living in an ecosystem 
sent3: humans eat crops 
sent4: controlling the amount of pollution is similar to reducing the amount of pollution 
sent5: polluting means something harmful / something poisonous is added in an environment causes harm in the environment 
sent6: a cause of something is a reason for that something 
sent7: waste has a negative impact on the environment 
sent8: fertilizers are a source of pollution 
sent9: adverse effect means negative impact 
sent10: water pollution is when humans pollute the environment with pollutants 
sent11: limiting a source of something tht has a negative impact on something else has a positive impact on that something else 
sent12: polluting means something poisonous is added to an environment 
sent13: if something has a negative impact on something else then increasing the amount of that something has a negative impact on that something else 
sent14: if crops are sprayed with something then that something will be on the crop 's surface 
sent15: a pesticide is used for protecting plants by killing insects 
sent16: as the amount of a source of something decreases , the amount of that something will decrease 
sent17: overuse is similar to depletion 
sent18: pollution is a source of pollutants 
sent19: eating food that contains pesticides can have a negative impact on the health of an animal 
sent20: human waste often causes harm to an environment 
sent21: damage has a negative impact on a thing 
sent22: destroying something causes harm to that something 
sent23: pesticides can cause pollution 
sent24: poisoning an environment means poisoning the organisms in that environment 
sent25: crops are a kind of edible plant for eating
Proof: sent23 & sent8 -> hypothesis;

Hypothesis: the function of red blood cells is to carry oxygen
Context:
sent1: a function is an activity 
sent2: if an organism can do something , then that organism is able to do that something 
sent3: transporting is a kind of function 
sent4: red blood cells carry oxygen 
sent5: to bring means to transport 
sent6: take in oxygen means get oxygen into the blood 
sent7: to be used for something means to be required by that something 
sent8: to transport can mean to carry 
sent9: role means function 
sent10: if something is transported to something else then that something else receives that something 
sent11: if something requires something else then that something else is important to that something 
sent12: blood absorbs oxygen in the lungs 
sent13: oxygenated means having oxygen 
sent14: oxygen can be found in air 
sent15: transport means to move / to make travel 
sent16: controlling is a kind of function 
sent17: to carry means to move something 
sent18: the brain is a part of the body 
sent19: purpose means role 
sent20: to do a job means to perform a job 
sent21: to have a function is similar to to be responsible for 
sent22: to be involved in something means to have a role in something 
sent23: cells require oxygen for fuel 
sent24: the function of something is what that something is used to do 
sent25: a vehicle can be used for transferring
Proof: sent3 & sent8 -> int1: carrying is a kind of function; int1 & sent4 -> hypothesis;""",
}


def format_input(ex) -> str:
    context = re.sub(r"sent(?=\d+)", "\nsent", ex["context"])
    return f"Hypothesis: {ex['hypothesis']}\nContext:{context}\nProof:".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="code-davinci-002")
    parser.add_argument(
        "--prompt",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Which of the three prompts to use.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=15,
        help="Time gap (in seconds) between two consecutive requests. Only necessary for Codex. (Default: 15)",
    )
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl",
        help="Path to the validation data.",
    )
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.data_path)]
    openai.api_key = args.api_key

    for ex in tqdm(data):
        prompt = prompt_base[args.prompt] + "\n\n" + format_input(ex)
        response = openai.Completion.create(
            model=args.model,
            prompt=prompt,
            temperature=0,
            max_tokens=256,
            stop="-> hypothesis;",
        )
        sleep(args.sleep)
        proof = (response["choices"][0]["text"] + "-> hypothesis;").strip()
        print(f"$proof$ = {proof}")


if __name__ == "__main__":
    main()
