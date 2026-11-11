# CPH 100 Final Project

The final project is an opportunity to investigate a question in machine learning for healthcare that interests you. You will work **with a partner** to define a focused question, investigate it using appropriate evidence or methods, and communicate your findings in a written report and an in-class poster session.

## Important dates

| Date | Milestone | Graded? |
| --- | --- | --- |
| **November 6** | Progress report due | No |
| **December 3** | Final project write-up due | Yes |
| **December 4** | In-class poster session (last day of class) | Yes |

## Partners

Students are expected to work **in pairs**. Please find a partner early so that you can develop a project idea together. You may use the course discussion board to find a partner. If you need help finding someone, email **Professor Chen** (iychen@berkeley.edu), who can help match students.

## Project scope

Choose a specific, feasible question related to machine learning and healthcare. Your project might involve implementing and evaluating a model, reproducing or extending a published result, analyzing a publicly available dataset, comparing methods or evaluation approaches, or critically examining evidence about a deployed clinical AI application. The strongest projects will go beyond summarizing existing work to make a focused contribution of their own, such as an empirical analysis, a careful comparison, or a well-supported original argument.

Be thoughtful about feasibility. A narrowly scoped project with a clear research question and convincing evaluation is preferable to an ambitious project that cannot be completed during the semester. If you plan to work with data, make sure it is accessible and appropriate to use; do not rely on obtaining restricted clinical data for your project.

The project should roughly be the length and depth of one of the problem sets of the class. Think of this as the fourth problem set. 

## November 6 — Progress report (not graded)

Submit a brief progress report describing your selected project and your plan for completing it. The purpose is to give the teaching staff an opportunity to **evaluate scope and provide guidance** before you invest substantial time in the project.

Your report should identify your partner, the problem or research question, why it matters, your proposed methods or approach, the data and other resources you expect to use (if applicable), how you plan to evaluate your work, and any foreseeable risks or open questions. Include a few relevant references. The progress report is **not graded**; the teaching staff will use it to provide feedback.

## December 3 — Final project write-up

Submit a written report (6 pages max) presenting your completed project. The report should explain the motivation and research question; situate your work in relevant prior literature; describe your methods, data, and evaluation approach as applicable; present your findings; and discuss limitations and implications. Use figures or tables where they help readers assess your conclusions, and cite relevant sources. Clearly distinguish results that you produced from claims taken from prior work.

## December 4 — In-class poster session

The poster session will take place **during class on December 4, the last day of class**. Each pair will prepare a poster that communicates the research question, approach, main findings, and implications of the project. Be prepared to explain your work and discuss it with classmates and teaching staff.

## Grading criteria

The final project will be evaluated on four criteria:

- **Clarity:** Is the research question clear? Are the methods, findings, and conclusions understandable and well organized?
- **Originality:** Does the project contribute an original analysis, evaluation, extension, synthesis, or perspective rather than simply restating existing work?
- **Significance:** Does the project address a meaningful question in healthcare or clinical AI, and explain why its results matter?
- **Quality:** Are the methods, evidence, analysis, and conclusions appropriate, careful, and well supported? Are limitations acknowledged?

These criteria apply to the final project as communicated through the write-up and poster.

## Example project ideas

The following are **starting points, not an exhaustive list**. You are welcome to propose a different topic. The cited papers are suggestions for beginning your investigation; you should explore additional relevant research. The examples below are adapted from topics in the CPH 200C graduate course.

1. **Evaluating AI scribes.** Compare approaches for measuring whether AI-generated clinical documentation improves clinician efficiency or reduces documentation burden. You could investigate how note quality and potential transcription errors should be evaluated. Although data for this application might be difficult to acquire, you could simulate a pipeline or an evaluation workflow. Starting points: Liu et al. (2024); Misurac et al. (2025); Koenecke et al. (2024).

2. **Clinical decision support in practice.** Study when an AI-generated alert or triage recommendation is likely to improve care. You might reproduce an evaluation on an accessible dataset, analyze failure modes, or compare model metrics with clinically meaningful outcomes. Starting points: Han et al. (2024); Lin et al. (2024); Savage et al. (2024).

3. **Clinicians working with AI.** Investigate the difference between model-alone performance and clinician-with-model performance. For example, design a small controlled comparison of human judgments with and without model advice, or analyze existing study designs and their limitations. Starting points: Goh et al. (2024); Eisemann et al. (2025); Tu et al. (2025).

4. **Cost-effectiveness of clinical AI.** Analyze the assumptions under which a clinical AI system saves money or improves outcomes cost-effectively. A project might implement a simple sensitivity analysis using values reported in the literature. Starting points: Hill et al. (2024); Thao et al. (2024); Wu et al. (2024).

5. **AI in health insurance decisions.** Investigate the design and consequences of algorithms used in insurance coverage decisions. You could analyze how to assess errors, incentives, accountability, or the evidence available to patients and clinicians. Starting points: Mello and Rose (2024); Rosenbluth (2024).

6. **Clinical LLM benchmarks.** Compare two evaluation approaches for medical LLMs, focusing on what each measures and what it misses. You might construct a small, carefully designed benchmark subset, compare prompts or grading criteria, or study sensitivity to test-set composition. Starting points: Arora et al. (2025); Bedi et al. (2026).

7. **FDA guidance and clinical AI.** Examine how regulatory guidance affects the design or evaluation of AI-enabled clinical software. Choose a specific type of system and analyze how it might be classified and what evidence would be needed to support its use. Starting point: the FDA's 2026 guidance on low-risk devices and clinical decision support software, discussed in the graduate-course problem set.

8. **Automation bias and deskilling.** Explore how repeated exposure to AI recommendations may affect clinicians' independent performance. You might propose an evaluation protocol or critically compare how existing studies measure reliance on AI. Starting points: Budzyń et al. (2025); Qazi et al. (2025).

9. **Extend a paper from class.** Reproduce a result from a course reading and test a focused extension, such as a new population, a different evaluation metric, a modified model, or a distribution shift. Select a paper with accessible data or code and define in advance what would count as a successful reproduction.

10. **AI in a clinical specialty.** Select an area such as radiology, oncology, or dementia care and investigate one concrete problem where AI might help. Identify available datasets and evaluation challenges, and—where feasible—run an empirical comparison or structured error analysis.

**Example projects from last year:** Professor Chen will upload examples of strong projects from last year to bCourses to help illustrate possible project scope and presentation approaches.

## References for the example ideas

- Arora, R. K., et al. (2025). *HealthBench: Evaluating large language models towards improved human health*. arXiv:2505.08775.
- Bedi, S., et al. (2026). *HealthAdminBench: Evaluating computer-use agents on healthcare administration tasks*. arXiv:2604.09937.
- Budzyń, K., et al. (2025). Endoscopist deskilling risk after exposure to artificial intelligence in colonoscopy: A multicentre observational study. *The Lancet Gastroenterology & Hepatology, 10*(10), 896–903.
- Eisemann, N., et al. (2025). Nationwide real-world implementation of AI for cancer detection in population-based mammography screening. *Nature Medicine*.
- Goh, E., et al. (2024). Large language model influence on diagnostic reasoning: A randomized clinical trial. *JAMA Network Open, 7*(10), e2440969.
- Han, R., et al. (2024). Randomised controlled trials evaluating artificial intelligence in clinical practice: A scoping review. *The Lancet Digital Health, 6*(5), e367–e373.
- Hill, H., et al. (2024). Cost-effectiveness of AI for risk-stratified breast cancer screening. *JAMA Network Open, 7*(9), e2431715.
- Koenecke, A., et al. (2024). Careless whisper: Speech-to-text hallucination harms. *ACM Conference on Fairness, Accountability, and Transparency*, 1672–1681.
- Lin, C.-S., et al. (2024). AI-enabled electrocardiography alert intervention and all-cause mortality: A pragmatic randomized clinical trial. *Nature Medicine, 30*(5), 1461–1470.
- Liu, T.-L., et al. (2024). Does AI-powered clinical documentation enhance clinician efficiency? A longitudinal study. *NEJM AI, 1*(12), AIoa2400659.
- Mello, M. M., & Rose, S. (2024). Denial—Artificial intelligence tools and health insurance coverage decisions. *JAMA Health Forum, 5*, e240622.
- Misurac, J., Knake, L. A., & Blum, J. M. (2025). The effect of ambient artificial intelligence notes on provider burnout. *Applied Clinical Informatics, 16*(2), 252–258.
- Qazi, I. A., et al. (2025). Automation bias in large language model assisted diagnostic reasoning among AI-trained physicians. *medRxiv*.
- Rosenbluth, T. (2024). [In constant battle with insurers, doctors reach for a cudgel: A.I.](https://www.nytimes.com/2024/07/10/health/doctors-insurers-artificial-intelligence.html). *The New York Times*.
- Savage, C. H., et al. (2024). Prospective evaluation of artificial intelligence triage of intracranial hemorrhage on noncontrast head CT examinations. *American Journal of Roentgenology, 223*(5), e2431639.
- Thao, V., et al. (2024). Cost-effectiveness of artificial intelligence-enabled electrocardiograms for early detection of low ejection fraction. *Mayo Clinic Proceedings: Digital Health, 2*(4), 620–631.
- Tu, T., et al. (2025). Towards conversational diagnostic artificial intelligence. *Nature*.
- Wu, X., et al. (2024). Cost-effectiveness and cost-utility of a digital technology-driven hierarchical healthcare screening pattern in China. *Nature Communications, 15*, 3650.
