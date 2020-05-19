import datetime
import os
import logging
import multiprocessing
import sys

import deap
from deap import base
from deap import creator
from deap import tools

import numpy as np
import pypet as pp
import pandas as pd

from ...utils import paths as paths
from ...utils import pypetUtils as pu

from . import evolutionaryUtils as eu
from . import deapUtils as du


class Evolution:
    """Evolutionary parameter optimization. This class helps you to optimize any function or model using an evlutionary algorithm. 
    It uses the package `deap` and supports its builtin mating and selection functions as well as custom ones. 
    """

    def __init__(
        self,
        evalFunction,
        parameterSpace,
        weightList=None,
        model=None,
        filename="evolution.hdf",
        ncores=None,
        POP_INIT_SIZE=100,
        POP_SIZE=20,
        NGEN=10,
        matingOperator=None,
        MATE_P=None,
        mutationOperator=None,
        MUTATE_P=None,
        selectionOperator=None,
        SELECT_P=None,
        parentSelectionOperator=None,
        PARENT_SELECT_P=None,
        individualGenerator=None,
        IND_GENERATOR_P=None
    ):
        """Initialize evolutionary optimization.
        :param evalFunction: Evaluation function of a run that provides a fitness vector and simulation outputs
        :type evalFunction: function
        :param parameterSpace: Parameter space to run evolution in.
        :type parameterSpace: `neurolib.utils.parameterSpace.ParameterSpace`
        :param weightList: List of floats that defines the dimensionality of the fitness vector returned from evalFunction and the weights of each component for multiobjective optimization (positive = maximize, negative = minimize). If not given, then a single positive weight will be used, defaults to None
        :type weightList: list[float], optional
        :param model: Model to simulate, defaults to None
        :type model: `neurolib.models.model.Model`, optional

        :param filename: HDF file to store all results in, defaults to "evolution.hdf"
        :type filename: str, optional
        :param ncores: Number of cores to simulate on (max cores default), defaults to None
        :type ncores: int, optional

        :param POP_INIT_SIZE: Size of first population to initialize evolution with (random, uniformly distributed), defaults to 100
        :type POP_INIT_SIZE: int, optional
        :param POP_SIZE: Size of the population during evolution, defaults to 20
        :type POP_SIZE: int, optional
        :param NGEN: Numbers of generations to evaluate, defaults to 10
        :type NGEN: int, optional

        :param matingOperator: Custom mating operator, defaults to deap.tools.cxBlend
        :type matingOperator: deap operator, optional
        :param MATE_P: Mating operator keyword arguments (for the default crossover operator cxBlend, this defaults `alpha` = 0.5)
        :type MATE_P: dict, optional

        :param mutationOperator: Custom mutation operator, defaults to du.gaussianAdaptiveMutation_nStepSizes
        :type mutationOperator: deap operator, optional
        :param MUTATE_P: Mutation operator keyword arguments
        :type MUTATE_P: dict, optional

        :param selectionOperator: Custom selection operator, defaults to du.selBest_multiObj
        :type selectionOperator: deap operator, optional
        :param SELECT_P: Selection operator keyword arguments
        :type SELECT_P: dict, optional

        :param parentSelectionOperator: Operator for parent selection, defaults to du.selRank
        :param PARENT_SELECT_P: Parent selection operator keyword arguments (for the default operator selRank, this defaults to `s` = 1.5 in Eiben&Smith p.81)
        :type PARENT_SELECT_P: dict, optional

        :param individualGenerator: Function to generate initial individuals, defaults to du.randomParametersAdaptive     
        """

        if weightList is None:
            logging.info("weightList not set, assuming single fitness value to be maximized.")
            weightList = [1.0]

        trajectoryName = "results" + datetime.datetime.now().strftime("-%Y-%m-%d-%HH-%MM-%SS")
        logging.info(f"Trajectory Name: {trajectoryName}")
        self.HDF_FILE = os.path.join(paths.HDF_DIR, filename)
        trajectoryFileName = self.HDF_FILE

        logging.info("Storing data to: {}".format(trajectoryFileName))
        logging.info("Trajectory Name: {}".format(trajectoryName))
        if ncores is None:
            ncores = multiprocessing.cpu_count()
        logging.info("Number of cores: {}".format(ncores))

        # initialize pypet environment
        # env = pp.Environment(trajectory=trajectoryName, filename=trajectoryFileName)
        env = pp.Environment(
            trajectory=trajectoryName,
            filename=trajectoryFileName,
            use_pool=False,
            multiproc=True,
            ncores=ncores,
            complevel=9,
            log_config=paths.PYPET_LOGGING_CONFIG,            
        )

        # Get the trajectory from the environment
        traj = env.traj
        # Sanity check if everything went ok
        assert (
            trajectoryName == traj.v_name
        ), f"Pypet trajectory has a different name than trajectoryName {trajectoryName}"
        # trajectoryName = traj.v_name

        self.model = model
        self.evalFunction = evalFunction
        self.weightList = weightList


        self.NGEN = NGEN
        assert POP_SIZE % 2 == 0, "Please chose an even number for POP_SIZE!"
        self.POP_SIZE = POP_SIZE
        assert POP_INIT_SIZE % 2 == 0, "Please chose an even number for POP_INIT_SIZE!"
        self.POP_INIT_SIZE = POP_INIT_SIZE
        self.ncores = ncores

        # comment string for storing info
        self.comments = "no comments"

        self.traj = env.traj
        self.env = env
        self.trajectoryName = trajectoryName
        self.trajectoryFileName = trajectoryFileName

        self._initialPopulationSimulated = False

        # -------- settings
        self.verbose = False
        self.plotColor = "C0"

        # -------- simulation
        self.parameterSpace = parameterSpace
        self.ParametersInterval = parameterSpace.named_tuple_constructor
        self.paramInterval = parameterSpace.named_tuple

        self.toolbox = deap.base.Toolbox()

        # register evolution operators 
        self.matingOperator = matingOperator or tools.cxBlend
        # default parameters for tools.cxBlend:
        if self.matingOperator == tools.cxBlend and MATE_P is None:
            MATE_P = {"alpha" : 0.5}
        self.MATE_P = MATE_P or {}

        self.mutationOperator = mutationOperator or du.gaussianAdaptiveMutation_nStepSizes
        self.MUTATE_P = MUTATE_P or {}

        self.selectionOperator = selectionOperator or du.selBest_multiObj
        self.SELECT_P = SELECT_P or {}

        self.parentSelectionOperator = parentSelectionOperator or du.selRank
        # default parameters for du.selRank:
        if self.parentSelectionOperator == du.selRank and PARENT_SELECT_P is None:
            PARENT_SELECT_P = {"s" : 1.5}
        self.PARENT_SELECT_P = PARENT_SELECT_P or {}

        self.individualGenerator = individualGenerator or du.randomParametersAdaptive        

        self.initDEAP(
            self.toolbox,
            self.env,
            self.paramInterval,
            self.evalFunction,
            weightList=self.weightList,
            matingOperator=self.matingOperator,
            mutationOperator=self.mutationOperator,
            selectionOperator=self.selectionOperator,
            parentSelectionOperator=self.parentSelectionOperator,
            individualGenerator=self.individualGenerator
        )

        # set up pypet trajectory
        self.initPypetTrajectory(
            self.traj, self.paramInterval, self.POP_SIZE, self.NGEN, self.model,
        )

        # population history: dict of all valid individuals per generation
        self.history = {}

        # initialize population
        self.evaluationCounter = 0
        self.last_id = 0

    def run(self, verbose=False):
        """Run the evolution or continue previous evolution. If evolution was not initialized first
        using `runInitial()`, this will be done.
        
        :param verbose: Print and plot state of evolution during run, defaults to False
        :type verbose: bool, optional
        """
        
        self.verbose = verbose
        if not self._initialPopulationSimulated:
            self.runInitial()

        self.runEvolution()

    def getIndividualFromTraj(self, traj):
        """Get individual from pypet trajectory
        
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :return: Individual (`DEAP` type)
        :rtype: `deap.creator.Individual`
        """
        # either pass an individual or a pypet trajectory with the attribute individual
        if type(traj).__name__ == "Individual":
            individual = traj
        else:
            individual = traj.individual
            ind_id = traj.id
            individual = [p for p in self.pop if p.id == ind_id]
            if len(individual) > 0:
                individual = individual[0]
        return individual

    def getModelFromTraj(self, traj):
        """Return the appropriate model with parameters for this individual
        :params traj: Pypet trajectory with individual (traj.individual) or directly a deap.Individual

        :returns model: Model with the parameters of this individual.
        
        :param traj: Pypet trajectory with individual (traj.individual) or directly a deap.Individual
        :type traj: `pypet.trajectory.Trajectory`
        :return: Model with the parameters of this individual.
        :rtype: `neurolib.models.model.Model`
        """
        model = self.model
        model.params.update(self.individualToDict(self.getIndividualFromTraj(traj)))
        return model

    def individualToDict(self, individual):
        """Convert an individual to a parameter dictionary.
        
        :param individual: Individual (`DEAP` type)
        :type individual: `deap.creator.Individual`
        :return: Parameter dictionary of this individual
        :rtype: dict
        """
        return self.ParametersInterval(*(individual[: len(self.paramInterval)]))._asdict().copy()

    def initPypetTrajectory(self, traj, paramInterval, POP_SIZE, NGEN, model):
        """Initializes pypet trajectory and store all simulation parameters for later analysis.
        
        :param traj: Pypet trajectory (must be already initialized!)
        :type traj: `pypet.trajectory.Trajectory`
        :param paramInterval: Parameter space, from ParameterSpace class
        :type paramInterval: parameterSpace.named_tuple
        :param POP_SIZE: Population size
        :type POP_SIZE: int
        :param MATE_P: Crossover parameter
        :type MATE_P: float
        :param NGEN: Number of generations
        :type NGEN: int
        :param model: Model to store the default parameters of
        :type model: `neurolib.models.model.Model`
        """
        # Initialize pypet trajectory and add all simulation parameters
        traj.f_add_parameter("popsize", POP_SIZE, comment="Population size")  #
        traj.f_add_parameter("NGEN", NGEN, comment="Number of generations")

        # Placeholders for individuals and results that are about to be explored
        traj.f_add_parameter("generation", 0, comment="Current generation")

        traj.f_add_result("scores", [], comment="Score of all individuals for each generation")
        traj.f_add_result_group("evolution", comment="Contains results for each generation")
        traj.f_add_result_group("outputs", comment="Contains simulation results")

        #TODO: save evolution parameters and operators as well, MATE_P, MUTATE_P, etc..

        # if a model was given, save its parameters
        # NOTE: Convert model.params to dict() since it is a dotdict() and pypet doesn't like that
        if model is not None:
            traj.f_add_result("params", dict(model.params), comment="Default parameters")

        # todo: initialize this after individuals have been defined!
        traj.f_add_parameter("id", 0, comment="Index of individual")
        traj.f_add_parameter("ind_len", 20, comment="Length of individual")
        traj.f_add_derived_parameter(
            "individual", [0 for x in range(traj.ind_len)], "An indivudal of the population",
        )

    def initDEAP(
        self, toolbox, pypetEnvironment, paramInterval, evalFunction, weightList, matingOperator, mutationOperator, selectionOperator, parentSelectionOperator, individualGenerator
    ):
        """Initializes DEAP and registers all methods to the deap.toolbox
        
        :param toolbox: Deap toolbox
        :type toolbox: deap.base.Toolbox
        :param pypetEnvironment: Pypet environment (must be initialized first!)
        :type pypetEnvironment: [type]
        :param paramInterval: Parameter space, from ParameterSpace class
        :type paramInterval: parameterSpace.named_tuple
        :param evalFunction: Evaluation function
        :type evalFunction: function
        :param weightList: List of weiths for multiobjective optimization
        :type weightList: list[float]
        :param matingOperator: Mating function (crossover)
        :type matingOperator: function
        :param selectionOperator: Parent selection function
        :type selectionOperator: function
        :param individualGenerator: Function that generates individuals
        """
        # ------------- register everything in deap
        deap.creator.create("FitnessMulti", deap.base.Fitness, weights=tuple(weightList))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMulti)

        # initially, each individual has randomized genes
        # need to create a lambda funciton because du.generateRandomParams wants an argument but
        # toolbox.register cannot pass an argument to it.
        toolbox.register(
            "individual",
            deap.tools.initIterate,
            deap.creator.Individual,
            lambda: individualGenerator(paramInterval),
        )
        logging.info(f"Evolution: Individual generation: {individualGenerator}")

        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
        toolbox.register("map", pypetEnvironment.run)
        toolbox.register("evaluate", evalFunction)
        toolbox.register("run_map", pypetEnvironment.run_map)        

        # Operator registering
        
        toolbox.register("mate", matingOperator)
        logging.info(f"Evolution: Mating operator: {matingOperator}")

        toolbox.register("mutate", mutationOperator)
        logging.info(f"Evolution: Mutation operator: {mutationOperator}")

        toolbox.register("selBest", du.selBest_multiObj)
        toolbox.register("selectParents", parentSelectionOperator)
        logging.info(f"Evolution: Parent selection: {parentSelectionOperator}")
        toolbox.register("select", selectionOperator)
        logging.info(f"Evolution: Selection operator: {selectionOperator}")


    def evalPopulationUsingPypet(self, traj, toolbox, pop, gIdx):
        """Evaluate the fitness of the popoulation of the current generation using pypet
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :param toolbox: `deap` toolbox
        :type toolbox: deap.base.Toolbox
        :param pop: Population
        :type pop: list
        :param gIdx: Index of the current generation
        :type gIdx: int
        :return: Evaluated population with fitnesses
        :rtype: list
        """
        # Add as many explored runs as individuals that need to be evaluated.
        # Furthermore, add the individuals as explored parameters.
        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`:
        # This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(
            pp.cartesian_product(
                {"generation": [gIdx], "id": [x.id for x in pop], "individual": [list(x) for x in pop],},
                [("id", "individual"), "generation"],
            )
        )  # the current generation  # unique id of each individual

        # increment the evaluationCounter
        self.evaluationCounter += len(pop)

        # run simulations for one generation
        evolutionResult = toolbox.map(toolbox.evaluate)

        # This error can have different reasons but is most likely
        # due to multiprocessing problems. One possibility is that your evaluation
        # funciton is not pickleable or that it returns an object that is not pickleable.
        assert len(evolutionResult) > 0, "No results returned from simulations."

        for idx, result in enumerate(evolutionResult):
            runIndex, packedReturnFromEvalFunction = result

            # packedReturnFromEvalFunction is the return from the evaluation function
            # it has length two, the first is the fitness, second is the model output
            assert (
                len(packedReturnFromEvalFunction) == 2
            ), "Evaluation function must return tuple with shape (fitness, output_data)"

            fitnessesResult, returnedOutputs = packedReturnFromEvalFunction

            # store simulation outputs
            pop[idx].outputs = returnedOutputs

            # store fitness values
            pop[idx].fitness.values = fitnessesResult

            # compute score
            pop[idx].fitness.score = np.ma.masked_invalid(pop[idx].fitness.wvalues).sum() / (len(pop[idx].fitness.wvalues))
        return pop

    def getValidPopulation(self, pop):
        return [p for p in pop if not (np.isnan(p.fitness.values).any() or np.isinf(p.fitness.values).any()) ]

    def getInvalidPopulation(self, pop):
        return [p for p in pop if np.isnan(p.fitness.values).any() or np.isinf(p.fitness.values).any()]

    def tagPopulation(self, pop):
        """Take a fresh population and add id's and attributes such as parameters that we can use later

        :param pop: Fresh population
        :type pop: list
        :return: Population with tags
        :rtype: list
        """
        for i, ind in enumerate(pop):
            assert not hasattr(ind, "id"), "Individual has an id already, will not overwrite it!"
            ind.id = self.last_id
            ind.gIdx = self.gIdx
            ind.simulation_stored = False
            ind_dict = self.individualToDict(ind)
            for key, value in ind_dict.items():
                # set the parameters as attributes for easy access
                setattr(ind, key, value)
            ind.params = ind_dict
            # increment id counter
            self.last_id += 1
        return pop

    def runInitial(self):
        """Run the first round of evolution with the initial population of size `POP_INIT_SIZE`
        and select the best `POP_SIZE` for the following evolution. This needs to be run before `runEvolution()`
        """
        self._t_start_initial_population = datetime.datetime.now()

        # Create the initial population
        self.pop = self.toolbox.population(n=self.POP_INIT_SIZE)

        ### Evaluate the initial population
        logging.info("Evaluating initial population of size %i ..." % len(self.pop))
        self.gIdx = 0  # set generation index
        self.pop = self.tagPopulation(self.pop)

        # evaluate
        self.pop = self.evalPopulationUsingPypet(self.traj, self.toolbox, self.pop, self.gIdx)

        if self.verbose:
            eu.printParamDist(self.pop, self.paramInterval, self.gIdx)

        # save all simulation data to pypet
        self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)

        # reduce initial population to popsize
        self.pop = self.toolbox.select(self.pop, k=self.traj.popsize, **self.SELECT_P)

        self._initialPopulationSimulated = True

        # populate history for tracking
        self.history[self.gIdx] = self.pop #self.getValidPopulation(self.pop)

        self._t_end_initial_population = datetime.datetime.now()

    def runEvolution(self):
        """Run the evolutionary optimization process for `NGEN` generations.
        """
        # Start evolution
        logging.info("Start of evolution")
        self._t_start_evolution = datetime.datetime.now()
        for self.gIdx in range(self.gIdx + 1, self.gIdx + self.traj.NGEN):
            # ------- Weed out the invalid individuals and replace them by random new indivuals -------- #
            validpop = self.getValidPopulation(self.pop)
            # replace invalid individuals
            invalidpop = self.getInvalidPopulation(self.pop)
            
            logging.info("Replacing {} invalid individuals.".format(len(invalidpop)))
            newpop = self.toolbox.population(n=len(invalidpop))
            newpop = self.tagPopulation(newpop)

            # ------- Create the next generation by crossover and mutation -------- #
            ### Select parents using rank selection and clone them ###
            offspring = list(map(self.toolbox.clone, self.toolbox.selectParents(self.pop, self.POP_SIZE, **self.PARENT_SELECT_P)))
            


            ##### cross-over ####
            for i in range(1, len(offspring), 2):
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i], **self.MATE_P)
                # delete fitness inherited from parents
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
                del offspring[i - 1].fitness.wvalues, offspring[i].fitness.wvalues

                # assign parent IDs to new offspring
                offspring[i - 1].parentIds = offspring[i - 1].id, offspring[i].id
                offspring[i].parentIds = offspring[i - 1].id, offspring[i].id

                # delete id originally set from parents, needs to be deleted here!
                # will be set later in tagPopulation()
                del offspring[i - 1].id, offspring[i].id

            ##### Mutation ####
            # Apply mutation
            du.mutateUntilValid(offspring, self.paramInterval, self.toolbox, MUTATE_P=self.MUTATE_P)

            offspring = self.tagPopulation(offspring)

            # ------- Evaluate next generation -------- #

            self.pop = offspring + newpop
            self.evalPopulationUsingPypet(self.traj, self.toolbox, offspring + newpop, self.gIdx)

            # log individuals
            self.history[self.gIdx] = validpop + offspring + newpop  # self.getValidPopulation(self.pop)            

            # ------- Select surviving population -------- #

            # select next generation
            self.pop = self.toolbox.select(validpop + offspring + newpop, k=self.traj.popsize, **self.SELECT_P)

            # ------- END OF ROUND -------



            # save all simulation data to pypet
            self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)

            # select best individual for logging
            self.best_ind = self.toolbox.selBest(self.pop, 1)[0]

            # text log
            next_print = print if self.verbose else logging.info
            next_print("----------- Generation %i -----------" % self.gIdx)
            next_print("Best individual is {}".format(self.best_ind))
            next_print("Score: {}".format(self.best_ind.fitness.score))
            next_print("Fitness: {}".format(self.best_ind.fitness.values))
            next_print("--- Population statistics ---")

            # verbose output
            if self.verbose:
                self.info(plot=True, info=True)
                # # population summary
                # eu.printParamDist(self.pop, self.paramInterval, self.gIdx)
                # bestN = 5
                # print(f"Best {bestN} individuals:")
                # eu.printIndividuals(self.toolbox.selBest(self.pop, bestN), self.paramInterval)                
                # # plotting
                # eu.plotPopulation(
                #     self.pop, self.paramInterval, self.gIdx, plotScattermatrix=True, save_plots=self.trajectoryName
                # )

        logging.info("--- End of evolution ---")
        logging.info("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
        logging.info("--- End of evolution ---")

        self.traj.f_store()  # We switched off automatic storing, so we need to store manually
        self._t_end_evolution = datetime.datetime.now()

        self.buildEvolutionTree()

    def buildEvolutionTree(self):
        """Builds a genealogy tree that is networkx compatible.

        Plot the tree using:

            import matplotlib.pyplot as plt
            import networkx as nx
            from networkx.drawing.nx_pydot import graphviz_layout

            G = nx.DiGraph(evolution.tree)
            G = G.reverse()     # Make the graph top-down
            pos = graphviz_layout(G, prog='dot')
            plt.figure(figsize=(8, 8))
            nx.draw(G, pos, node_size=50, alpha=0.5, node_color=list(evolution.genx.values()), with_labels=False)
            plt.show()
        """
        self.tree = dict()
        self.id_genx = dict()
        self.id_score = dict()

        for gen, pop in self.history.items():
            for p in pop:
                self.tree[p.id] = p.parentIds if hasattr(p, "parentIds") else ()
                self.id_genx[p.id] = p.gIdx
                self.id_score[p.id] = p.fitness.score


    def info(self, plot=True, bestN=5, info=True):
        """Print and plot information about the evolution and the current population
        
        :param plot: plot a plot using `matplotlib`, defaults to True
        :type plot: bool, optional
        :param bestN: Print summary of `bestN` best individuals, defaults to 5
        :type bestN: int, optional
        :param info: Print information about the evolution environment
        :type info: bool, optional
        """
        if info:
            eu.printEvolutionInfo(self)
        validPop = self.getValidPopulation(self.pop)
        scores = self.getScores()
        # Text output
        print("--- Info summary ---")
        print("Valid: {}".format(len(validPop)))
        print("Mean score (weighted fitness): {:.2}".format(np.mean(scores)))
        eu.printParamDist(self.pop, self.paramInterval, self.gIdx)
        print("--------------------")
        print(f"Best {bestN} individuals:")
        eu.printIndividuals(self.toolbox.selBest(self.pop, bestN), self.paramInterval)
        print("--------------------")
        # Plotting
        if plot:
            # hack: during the evolution we need to use reverse=True
            # after the evolution (with evolution.info()), we need False
            self.plotProgress(reverse=info)
            eu.plotPopulation(self, plotScattermatrix=True, save_plots=self.trajectoryName, color=self.plotColor)

    def plotProgress(self, reverse=True):
        """Plots progress of fitnesses of current evolution run
        """
        eu.plotProgress(self, reverse=reverse)

    def loadEvolution(self, fname):
        import dill
        evolution = dill.load(open(fname, "rb"))
        evolution.__init__(lambda x: x, self.parameterSpace)
        return evolution

    @property
    def dfPop(self):
        """Returns a `pandas` dataframe of the current generation's population parameters 
        for post processing. This object can be further used to easily analyse the population.
        :return: Pandas dataframe with all individuals and their parameters
        :rtype: `pandas.core.frame.DataFrame`
        """
        validPop = self.getValidPopulation(self.pop)
        indIds = [p.id for p in validPop]
        popArray = np.array([p[0 : len(self.paramInterval._fields)] for p in validPop]).T
        scores = self.getScores()
        
        dfPop = pd.DataFrame(popArray, index=self.parameterSpace.parameterNames).T
        dfPop["score"] = scores
        dfPop["id"] = indIds

        # add fitness columns
        n_fitnesses = len(self.pop[0].fitness.values)
        for i in range(n_fitnesses):
            for ip, p in enumerate(self.pop):
                column_name = "f" + str(i)
                dfPop.loc[ip, column_name] = p.fitness.values[i]
        return dfPop

    def loadResults(self, filename=None, trajectoryName=None):
        """Load results from a hdf file of a previous evolution and store the
        pypet trajectory in `self.traj`
        
        :param filename: hdf filename of the previous run, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory in the hdf file to load. If not given, the last one will be loaded, defaults to None
        :type trajectoryName: str, optional
        """
        if filename == None:
            filename = self.HDF_FILE
        self.traj = pu.loadPypetTrajectory(filename, trajectoryName)

    def getScores(self):
        """Returns the scores of the current valid population
        """
        validPop = self.getValidPopulation(self.pop)
        return np.array([validPop[i].fitness.score for i in range(len(validPop))])

    def getScoresDuringEvolution(self, traj=None, drop_first=True, reverse=False):
        """Get the scores of each generation's population.
        
        :param traj: Pypet trajectory. If not given, the current trajectory is used, defaults to None
        :type traj: `pypet.trajectory.Trajectory`, optional
        :param drop_first: Drop the first (initial) generation. This can be usefull because it can have a different size (`POP_INIT_SIZE`) than the succeeding populations (`POP_SIZE`) which can make data handling tricky, defaults to True
        :type drop_first: bool, optional
        :param reverse: Reverse the order of each generation. This is a necessary workaraound because loading from the an hdf file returns the generations in a reversed order compared to loading each generation from the pypet trajectory in memory, defaults to False
        :type reverse: bool, optional
        :return: Tuple of list of all generations and an array of the scores of all individuals
        :rtype: tuple[list, numpy.ndarray]
        """
        if traj == None:
            traj = self.traj

        generation_names = list(traj.results.evolution.f_to_dict(nested=True).keys())

        if reverse:
            generation_names = generation_names[::-1]
        if drop_first:
            generation_names.remove("gen_000000")

        npop = len(traj.results.evolution[generation_names[0]].scores)

        gens = []
        all_scores = np.empty((len(generation_names), npop))

        for i, r in enumerate(generation_names):
            gens.append(i)
            scores = traj.results.evolution[r].scores
            all_scores[i] = scores

        if drop_first:
            gens = np.add(gens, 1)

        return gens, all_scores
