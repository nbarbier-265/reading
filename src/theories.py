from src.models import InterventionType

THEORY_DESCRIPTIONS_COMPLEX: dict[InterventionType, str] = {
    InterventionType.METACOGNITIVE: """
    Metacognitive interventions activate higher-order thinking about one's own learning processes.
    Building on Flavell's (1979) foundational work and expanded by contemporary research from
    Dunlosky & Metcalfe (2020), these interventions develop self-regulatory cognitive mechanisms
    that enhance learning outcomes through strategic awareness and reflection.

    For students developing reading comprehension, metacognitive interventions are particularly 
    valuable as shown by research from Veenman et al. (2006) demonstrating that explicit 
    metacognitive strategy instruction significantly improves reading outcomes across diverse 
    student populations and text types by helping students develop self-awareness of their 
    learning processes and strategies.
    
    Key characteristics:
    - Prompts self-reflection about comprehension strategies
    - Encourages identification of confusion points or knowledge gaps
    - Supports planning and goal-setting for learning
    - Develops awareness of when and how to apply different reading strategies

    Effective approaches can ask students to reflect on their own learning strategies, 
    provide examples of how to apply different reading strategies, encourage strategy variety, 
    help students recognize effective strategy use, and identify when strategies are being used ineffectively.
    
    Example triggers: "What strategy would help you understand this better?", "How would you explain 
    this concept to someone else?", "What questions do you still have about this topic?"
    """,
    InterventionType.CONCEPTUAL: """
    Conceptual interventions illuminate core principles and structural relationships within knowledge domains.
    Drawing from Bransford et al.'s (2000) work on knowledge frameworks and cognitive load theory
    research (Sweller et al., 2019), these interventions optimize mental model construction and
    facilitate schema development that enables efficient transfer across contexts.

    For students developing reading comprehension, conceptual interventions are particularly 
    valuable as they help readers organize information hierarchically, distinguish between 
    main ideas and supporting details, and construct coherent mental representations of text. 
    Research by Willingham (2017) demonstrates that students who develop strong conceptual 
    frameworks show significantly improved ability to comprehend complex texts, make inferences, 
    and transfer understanding across different reading contexts by helping them understand the 
    relationships between concepts and the underlying principles of the text.
    
    Key characteristics:
    - Highlights fundamental principles and big ideas
    - Clarifies relationships between concepts
    - Addresses common misconceptions
    - Connects new information to established knowledge frameworks

    Effective implementations include comparing and contrasting related concepts, explaining 
    cause-effect relationships, identifying patterns across examples, and visualizing abstract ideas.
    
    Example triggers: "What is the relationship between these concepts?", "How does this relate to 
    the other concepts in the text?", "What is the cause of this effect?", "What is the effect of 
    this cause?"
    """,
    InterventionType.APPLICATION: """
    Application interventions bridge theoretical knowledge and authentic implementation contexts.
    Synthesizing situated cognition theory (Brown, Collins & Duguid, 1989) with transfer science
    research, these interventions activate connections between abstract concepts and concrete
    experiences, enhancing motivation through relevance while strengthening memory consolidation.

    For students developing reading comprehension, application interventions are particularly 
    valuable as they transform abstract text into meaningful experiences. Research by Duke et al. (2021) 
    demonstrates that when students connect text content to real-world applications, they achieve 
    significantly higher comprehension scores, deeper retention, and increased engagement with 
    challenging material. These interventions help bridge the gap between decoding text and 
    constructing meaning by providing concrete anchors for abstract concepts encountered during 
    reading.
    
    Key characteristics:
    - Connects abstract concepts to concrete situations
    - Demonstrates practical relevance of information
    - Provides opportunities to apply knowledge in varied contexts
    - Builds transfer skills for using knowledge flexibly
    - Helps students see the connections between different parts of a story or text
    - Encourages students to make connections between new and prior knowledge
    
    Effective formats include real-world problems, scenarios that require applying concepts, 
    examples showing practical applications, and questions about how information could be used 
    in different situations.

    Example triggers: "How would you use this in real life?", "What are some real-world examples of 
    this concept?", "How does this apply to your own experiences?", "What are some practical ways to 
    apply this knowledge?"
    """,
    InterventionType.VOCABULARY: """
    Vocabulary interventions develop domain-specific linguistic frameworks essential for conceptual mastery.
    Extending Nagy & Townsend's (2012) academic language research with cognitive linguistics and
    psycholinguistic processing models (Perfetti & Stafura, 2014), these interventions support lexical
    network development within meaningful semantic contexts.

    For students developing reading comprehension, vocabulary interventions are particularly 
    valuable as they help bridge the gap between decoding text and constructing meaning. 
    Research by Cervetti & Wright (2020) shows that students who receive explicit support 
    in identifying and understanding key vocabulary while reading demonstrate 30-45% greater 
    comprehension gains compared to control groups. These interventions help readers 
    organize information hierarchically, distinguish between main ideas and supporting details, 
    and construct coherent mental representations of text. By highlighting conceptual frameworks, 
    these interventions reduce cognitive load during reading and enable deeper processing of complex information.
    
    Key characteristics:
    - Explains unfamiliar or technical terms in accessible language
    - Provides clear definitions with examples
    - Shows relationships between related terms
    - Builds word knowledge in meaningful contexts
    
    Effective approaches include explaining terms when first encountered, connecting new vocabulary 
    to familiar concepts, showing how terms are used in context, and highlighting word parts that 
    provide meaning clues.

    Example triggers: "What does this word mean?", "How is this word used in the text?", "What is the 
    relationship between these words?", "How does this word make you feel?", "What is the history of 
    this word?", "What is the origin of this word?"
    """,
}

THEORY_DESCRIPTIONS_SIMPLE: dict[InterventionType, str] = {
    InterventionType.METACOGNITIVE: """
    Metacognitive interventions help students think about their own learning and understanding.
    
    Key goals:
    - Get students to reflect on what they understand and don't understand
    - Help students monitor their own comprehension
    - Teach strategies for better understanding
    - Build self-awareness of learning process
    
    Example prompts:
    - "What do you already know about this topic?"
    - "What parts of this are confusing to you?"
    - "What strategy could you use to understand this better?"
    - "How would you explain this to someone else?"
    - "What questions do you still have?"
    """,
    InterventionType.CONCEPTUAL: """
    Conceptual interventions help students understand key ideas and how they connect.
    
    Key goals:
    - Identify main ideas and supporting details
    - Show relationships between concepts
    - Clear up misconceptions
    - Connect new ideas to what students already know
    
    Example prompts:
    - "What's the main idea here?"
    - "How does this connect to what we learned before?"
    - "Why do you think this happens?"
    - "What's the difference between these ideas?"
    """,
    InterventionType.APPLICATION: """
    Application interventions help students connect ideas to real life and use what they learn.
    
    Key goals:
    - Link concepts to real-world examples
    - Show why ideas are useful and relevant
    - Help students apply knowledge in new situations
    - Connect to students' own experiences
    
    Example prompts:
    - "When might you use this in real life?"
    - "What's an example of this from your experience?"
    - "How could this help you solve a problem?"
    - "Where else have you seen something like this?"
    """,
    InterventionType.VOCABULARY: """
    Vocabulary interventions help students understand and use important words.
    
    Key goals:
    - Explain unfamiliar words clearly
    - Show how words are used in context
    - Connect words to familiar concepts
    - Build vocabulary knowledge
    
    Example prompts:
    - "What do you think this word means?"
    - "Can you use this word in a sentence?"
    - "What's another word that means the same thing?"
    - "Where have you heard this word before?"
    """,
}

# Piagetian developmental framework for grade-appropriate interventions
PIAGETIAN_GRADE_FRAMEWORK: dict[int, dict[InterventionType, str]] = {
    # Grade K
    0: {
        InterventionType.METACOGNITIVE: """
        At kindergarten level, metacognition is emerging through basic awareness of thinking.
        Students can identify when they don't understand something simple and ask for help.
        Effective interventions use concrete examples and visual aids to help students
        reflect on what they know and don't know about familiar topics.
        """,
        InterventionType.CONCEPTUAL: """
        Kindergarteners understand concepts through direct experience and concrete examples.
        They can group similar objects and recognize basic patterns but struggle with abstraction.
        Effective interventions use physical objects, pictures, and simple categorization
        activities to build foundational conceptual understanding.
        """,
        InterventionType.APPLICATION: """
        Application for kindergarteners involves connecting story elements to their immediate
        personal experiences. They can recognize similarities between stories and their own lives.
        Effective interventions ask students to share personal connections and use role-play
        to apply story concepts to familiar situations.
        """,
        InterventionType.VOCABULARY: """
        Vocabulary development at kindergarten focuses on concrete nouns and action verbs
        related to everyday experiences. Students learn words through context, repetition,
        and visual associations. Effective interventions use pictures, gestures, and simple
        definitions with examples from the child's environment.
        """,
    },
    # Grade 1
    1: {
        InterventionType.METACOGNITIVE: """
        First graders begin to monitor their understanding of simple texts and can
        identify when something doesn't make sense. They benefit from explicit modeling
        of thinking processes. Effective interventions use think-alouds and simple
        self-questioning techniques focused on comprehension monitoring.
        """,
        InterventionType.CONCEPTUAL: """
        First graders can understand basic cause-effect relationships and simple
        sequences. They still rely heavily on concrete examples but can begin making
        basic inferences. Effective interventions use sequencing activities, simple
        prediction tasks, and visual organizers to build conceptual understanding.
        """,
        InterventionType.APPLICATION: """
        First graders can apply story lessons to their own experiences and recognize
        basic similarities between text scenarios and real life. Effective interventions
        encourage students to share personal connections and guide them to recognize
        how story elements relate to their daily experiences.
        """,
        InterventionType.VOCABULARY: """
        First grade vocabulary development expands to include more descriptive words
        and basic academic terms. Students benefit from contextual learning and word
        categorization activities. Effective interventions use word-picture matching,
        simple definitions, and opportunities to use new words in familiar contexts.
        """,
    },
    # Grade 2
    2: {
        InterventionType.METACOGNITIVE: """
        Second graders develop greater awareness of their thinking and can identify
        specific parts of text they don't understand. They can use simple fix-up
        strategies when prompted. Effective interventions teach basic comprehension
        monitoring and introduce simple strategies like rereading and using pictures.
        """,
        InterventionType.CONCEPTUAL: """
        Second graders begin to understand more complex relationships between ideas
        and can identify main ideas with supporting details. They still benefit from
        concrete examples but can handle more abstract connections. Effective interventions
        use graphic organizers and guided discussions about relationships between concepts.
        """,
        InterventionType.APPLICATION: """
        Second graders can apply story concepts to situations beyond their immediate
        experience and begin to recognize broader patterns. Effective interventions
        guide students to connect text to both personal experiences and new situations,
        encouraging them to explain their reasoning.
        """,
        InterventionType.VOCABULARY: """
        Second grade vocabulary work expands to include more abstract terms and academic
        language. Students benefit from explicit instruction in word relationships and
        morphology. Effective interventions include word sorts, simple context clues
        instruction, and opportunities to use new vocabulary in writing and speaking.
        """,
    },
    # Grade 3
    3: {
        InterventionType.METACOGNITIVE: """
        Third graders can monitor comprehension more independently and identify specific
        comprehension problems. They can select appropriate strategies with guidance.
        Effective interventions teach multiple fix-up strategies and encourage students
        to explain their thinking process when solving comprehension problems.
        """,
        InterventionType.CONCEPTUAL: """
        Third graders understand more complex relationships between ideas and can
        identify implicit connections. They can organize information into categories
        and hierarchies. Effective interventions use compare/contrast activities,
        concept mapping, and guided discussions about text structure and organization.
        """,
        InterventionType.APPLICATION: """
        Third graders can apply concepts to hypothetical situations and begin to
        generalize learning across contexts. Effective interventions encourage students
        to generate examples of how concepts might apply in new situations and to
        explain the reasoning behind their applications.
        """,
        InterventionType.VOCABULARY: """
        Third grade vocabulary development includes more abstract terms, academic
        vocabulary, and domain-specific words. Students benefit from instruction in
        context clues and word parts. Effective interventions include semantic mapping,
        word analysis activities, and opportunities to use new vocabulary in meaningful contexts.
        """,
    },
    # Grade 4
    4: {
        InterventionType.METACOGNITIVE: """
        Fourth graders can monitor understanding across longer texts and identify
        specific strategies for different comprehension problems. They benefit from
        reflecting on strategy effectiveness. Effective interventions teach strategy
        selection and encourage students to evaluate which strategies work best for them.
        """,
        InterventionType.CONCEPTUAL: """
        Fourth graders understand complex relationships between ideas and can identify
        themes and patterns across texts. They can organize information into more
        sophisticated hierarchies. Effective interventions use thematic analysis,
        complex graphic organizers, and discussions about author's purpose and perspective.
        """,
        InterventionType.APPLICATION: """
        Fourth graders can apply concepts to novel situations and begin to understand
        how principles generalize across domains. Effective interventions encourage
        students to identify real-world applications of text concepts and to explain
        how these applications might vary in different contexts.
        """,
        InterventionType.VOCABULARY: """
        Fourth grade vocabulary work includes figurative language, multiple-meaning words,
        and specialized academic vocabulary. Students benefit from instruction in word
        relationships and connotations. Effective interventions include analyzing word
        networks, exploring nuances of meaning, and using new vocabulary in analytical writing.
        """,
    },
    # Grade 5
    5: {
        InterventionType.METACOGNITIVE: """
        Fifth graders can plan their approach to reading tasks, monitor comprehension
        independently, and adjust strategies as needed. They can reflect on their
        learning process. Effective interventions focus on self-regulation of reading
        and encourage students to set goals and evaluate their progress.
        """,
        InterventionType.CONCEPTUAL: """
        Fifth graders understand abstract concepts and can analyze relationships
        between ideas across texts. They can identify underlying assumptions and
        principles. Effective interventions use analytical discussions, complex
        text comparisons, and activities that require students to identify conceptual frameworks.
        """,
        InterventionType.APPLICATION: """
        Fifth graders can apply concepts to diverse contexts and understand how
        principles might need to be adapted in different situations. Effective
        interventions encourage students to analyze how concepts apply differently
        across contexts and to evaluate the effectiveness of different applications.
        """,
        InterventionType.VOCABULARY: """
        Fifth grade vocabulary development includes abstract concepts, technical terms,
        and nuanced academic language. Students benefit from instruction in etymology
        and connotation. Effective interventions include analyzing word origins,
        exploring shades of meaning, and using precise vocabulary in analytical contexts.
        """,
    },
    # Grade 6
    6: {
        InterventionType.METACOGNITIVE: """
        Sixth graders can analyze their own thinking processes and identify patterns
        in their comprehension strengths and challenges. They benefit from metacognitive
        discussions. Effective interventions focus on developing personalized strategy
        systems and encourage students to articulate their cognitive processes.
        """,
        InterventionType.CONCEPTUAL: """
        Sixth graders understand complex and abstract conceptual frameworks and can
        analyze how concepts relate within systems. They can identify underlying
        principles across domains. Effective interventions use systems thinking
        activities, conceptual analysis, and discussions about theoretical frameworks.
        """,
        InterventionType.APPLICATION: """
        Sixth graders can apply concepts to complex, multi-variable situations and
        understand how principles interact with contextual factors. Effective
        interventions encourage students to analyze real-world applications with
        multiple variables and to evaluate the effectiveness of different approaches.
        """,
        InterventionType.VOCABULARY: """
        Sixth grade vocabulary work includes abstract academic language, domain-specific
        terminology, and words with complex morphological structures. Students benefit
        from instruction in word relationships within conceptual frameworks. Effective
        interventions include analyzing technical vocabulary, exploring conceptual
        word networks, and using precise academic language in disciplinary contexts.
        """,
    },
    # Grade 7
    7: {
        InterventionType.METACOGNITIVE: """
        Seventh graders can analyze the effectiveness of different cognitive strategies
        and adapt their approach based on task demands. They benefit from metacognitive
        discussions about complex texts. Effective interventions focus on strategic
        reading across genres and encourage students to develop personalized systems
        for monitoring comprehension of challenging material.
        """,
        InterventionType.CONCEPTUAL: """
        Seventh graders understand abstract theoretical frameworks and can analyze
        relationships between concepts across disciplines. They can evaluate competing
        conceptual models. Effective interventions use interdisciplinary analysis,
        theoretical comparisons, and discussions about conceptual tensions and contradictions.
        """,
        InterventionType.APPLICATION: """
        Seventh graders can apply theoretical frameworks to complex situations and
        understand how principles might need to be modified based on contextual factors.
        Effective interventions encourage students to analyze how concepts apply in
        ambiguous situations and to evaluate the limitations of different applications.
        """,
        InterventionType.VOCABULARY: """
        Seventh grade vocabulary development includes specialized academic terminology,
        abstract concepts, and words with nuanced connotations. Students benefit from
        instruction in discipline-specific language. Effective interventions include
        analyzing technical vocabulary in context, exploring conceptual hierarchies,
        and using precise academic language in analytical writing.
        """,
    },
    # Grade 8
    8: {
        InterventionType.METACOGNITIVE: """
        Eighth graders can analyze their cognitive processes across different types
        of texts and tasks, identifying patterns in their approach. They benefit from
        metacognitive reflection on complex reading. Effective interventions focus on
        strategic adaptation to different text structures and encourage students to
        develop systems for approaching unfamiliar or challenging material.
        """,
        InterventionType.CONCEPTUAL: """
        Eighth graders understand complex theoretical models and can analyze relationships
        between abstract concepts. They can evaluate the strengths and limitations of
        different conceptual frameworks. Effective interventions use theoretical analysis,
        conceptual modeling, and discussions about how frameworks shape understanding.
        """,
        InterventionType.APPLICATION: """
        Eighth graders can apply abstract principles to complex real-world situations
        and understand how theoretical frameworks inform practical applications.
        Effective interventions encourage students to analyze how concepts apply in
        multifaceted situations and to evaluate the ethical implications of different applications.
        """,
        InterventionType.VOCABULARY: """
        Eighth grade vocabulary work includes technical terminology, abstract concepts,
        and words with complex etymological relationships. Students benefit from
        instruction in specialized academic language. Effective interventions include
        analyzing discipline-specific vocabulary, exploring conceptual networks,
        and using precise technical language in analytical contexts.
        """,
    },
    # Grade 9
    9: {
        InterventionType.METACOGNITIVE: """
        Ninth graders can analyze their cognitive processes across disciplines and
        develop personalized systems for approaching complex texts. They benefit from
        metacognitive discussions about disciplinary thinking. Effective interventions
        focus on discipline-specific reading strategies and encourage students to
        reflect on how different fields require different cognitive approaches.
        """,
        InterventionType.CONCEPTUAL: """
        Ninth graders understand sophisticated theoretical frameworks and can analyze
        how concepts function within disciplinary systems. They can evaluate competing
        paradigms. Effective interventions use paradigmatic analysis, theoretical
        critique, and discussions about how conceptual frameworks shape disciplinary knowledge.
        """,
        InterventionType.APPLICATION: """
        Ninth graders can apply theoretical frameworks to complex, ambiguous situations
        and understand how disciplinary concepts inform real-world applications.
        Effective interventions encourage students to analyze how abstract principles
        apply in multifaceted contexts and to evaluate the limitations of theoretical applications.
        """,
        InterventionType.VOCABULARY: """
        Ninth grade vocabulary development includes specialized disciplinary terminology,
        abstract theoretical concepts, and words with complex semantic relationships.
        Students benefit from instruction in discipline-specific discourse. Effective
        interventions include analyzing technical vocabulary in theoretical contexts,
        exploring disciplinary language conventions, and using precise academic language
        in specialized writing.
        """,
    },
    # Grade 10
    10: {
        InterventionType.METACOGNITIVE: """
        Tenth graders can analyze their cognitive processes across complex texts and
        tasks, developing sophisticated metacognitive systems. They benefit from
        discussions about epistemology. Effective interventions focus on epistemological
        reflection and encourage students to analyze how knowledge structures influence
        comprehension and interpretation.
        """,
        InterventionType.CONCEPTUAL: """
        Tenth graders understand complex theoretical systems and can analyze relationships
        between abstract concepts across disciplines. They can evaluate the philosophical
        foundations of different frameworks. Effective interventions use interdisciplinary
        analysis, theoretical critique, and discussions about the philosophical assumptions
        underlying conceptual models.
        """,
        InterventionType.APPLICATION: """
        Tenth graders can apply sophisticated theoretical frameworks to complex real-world
        situations and understand how abstract principles inform practical applications
        across contexts. Effective interventions encourage students to analyze how
        theoretical models apply in ambiguous situations and to evaluate the ethical
        and practical implications of different applications.
        """,
        InterventionType.VOCABULARY: """
        Tenth grade vocabulary work includes specialized theoretical terminology,
        abstract concepts with philosophical dimensions, and words with complex
        interdisciplinary relationships. Students benefit from instruction in the
        language of theoretical discourse. Effective interventions include analyzing
        philosophical terminology, exploring conceptual networks across disciplines,
        and using precise academic language in theoretical contexts.
        """,
    },
    # Grade 11
    11: {
        InterventionType.METACOGNITIVE: """
        Eleventh graders can analyze their cognitive processes across complex theoretical
        texts, developing sophisticated metacognitive systems. They benefit from
        epistemological discussions about knowledge construction. Effective interventions
        focus on critical reflection on knowledge paradigms and encourage students to
        analyze how different epistemological approaches influence understanding.
        """,
        InterventionType.CONCEPTUAL: """
        Eleventh graders understand sophisticated theoretical frameworks and can analyze
        how concepts function within and across knowledge systems. They can evaluate
        the philosophical and historical foundations of different paradigms. Effective
        interventions use paradigmatic analysis, theoretical critique, and discussions
        about how conceptual frameworks evolve and compete.
        """,
        InterventionType.APPLICATION: """
        Eleventh graders can apply complex theoretical frameworks to ambiguous real-world
        situations and understand how abstract principles inform practical applications
        across diverse contexts. Effective interventions encourage students to analyze
        how theoretical models apply in complex systems and to evaluate the ethical,
        social, and practical implications of different applications.
        """,
        InterventionType.VOCABULARY: """
        Eleventh grade vocabulary development includes specialized theoretical terminology,
        abstract concepts with philosophical dimensions, and words with complex historical
        and cultural connotations. Students benefit from instruction in the evolution
        of academic discourse. Effective interventions include analyzing theoretical
        vocabulary in historical context, exploring conceptual evolution across disciplines,
        and using precise academic language in sophisticated analytical contexts.
        """,
    },
    # Grade 12
    12: {
        InterventionType.METACOGNITIVE: """
        Twelfth graders can analyze their cognitive processes across complex theoretical
        texts and develop sophisticated metacognitive systems that integrate multiple
        epistemological perspectives. They engage in critical reflection on knowledge
        construction and validation. Effective interventions focus on epistemological
        analysis and encourage students to evaluate how different ways of knowing
        shape understanding and interpretation of complex texts.
        """,
        InterventionType.CONCEPTUAL: """
        Twelfth graders understand sophisticated theoretical frameworks and can analyze
        how concepts function within competing paradigms. They can evaluate the
        philosophical, historical, and cultural foundations of different knowledge systems.
        Effective interventions use meta-theoretical analysis, paradigmatic critique,
        and discussions about how conceptual frameworks reflect and shape cultural values.
        """,
        InterventionType.APPLICATION: """
        Twelfth graders can apply complex theoretical frameworks to ambiguous, multifaceted
        situations and understand how abstract principles inform practical applications
        across diverse contexts. They can analyze the limitations and contextual constraints
        of theoretical applications. Effective interventions encourage students to evaluate
        how theoretical models apply in complex systems with competing values and to
        analyze the ethical, social, and practical implications of different applications.
        """,
        InterventionType.VOCABULARY: """
        Twelfth grade vocabulary development includes specialized theoretical terminology,
        abstract concepts with philosophical dimensions, and words with complex historical,
        cultural, and disciplinary connotations. Students engage with the language of
        meta-theoretical discourse. Effective interventions include analyzing how
        terminology reflects epistemological assumptions, exploring conceptual evolution
        across intellectual traditions, and using precise academic language in
        sophisticated theoretical contexts.
        """,
    },
}
