# NN-ASSIGNMENT_5
name:B.KISHORE BABU
ID:700752976
1)Explain the adversarial process in GAN training. What are the goals of the generator and discriminator, and how do they improve through competition? Diagram of the GAN architecture showing the data flow and objectives of each component.

ans)

Generative Adversarial Networks (GANs) are a type of deep learning architecture composed of two neural networks — a Generator (G) and a Discriminator (D) — that are trained simultaneously in a competitive process.

Goals of Each Component:
Generator (G)
Goal: Create data that is so realistic it fools the discriminator.

Input: Random noise (latent vector) z sampled from a simple distribution (e.g., Gaussian).

Output: Synthetic data (e.g., an image).

Discriminator (D)
Goal: Accurately distinguish between real data (from the training set) and fake data (from the generator).

Input: Data sample (either real or generated).

Output: A probability score:

Close to 1 → real

Close to 0 → fake

Adversarial Training Process:
Generator training:

Tries to maximize the probability that the discriminator classifies its fake outputs as real.

Loss function: encourages the generator to improve based on how well it fools the discriminator.

Discriminator training:

Tries to maximize correct classification: real data as real, and fake data as fake.

Loss function: binary cross-entropy on both real and fake samples.

This zero-sum game continues:

The generator improves by generating more realistic data.

The discriminator improves by becoming better at spotting fakes.

Eventually, if training is successful, the generator becomes so good that the discriminator cannot distinguish real from fake — i.e., the discriminator's accuracy drops to 50%, indicating perfect deception
DIAGRAM

I AM NOT ABLE DO PICTURE HERE SO I DID IN MY PDF FILE CAN YIU PLEASE CHECK IT PROFESSOR

2) Ethics and AI Harm

Choose one of the following real-world AI harms discussed in Chapter 12:
Representational harm
Allocational harm
Misinformation in generative AI
Describe a real or hypothetical application where this harm may occur. Then, suggest two harm mitigation strategies that could reduce its impact based on the lecture.
ANS)
 Real-World Application: Facial Recognition Systems
A well-documented case of representational harm involves facial recognition systems that perform poorly on people with darker skin tones. Studies, such as the Gender Shades project, showed that commercial facial recognition systems had significantly higher error rates for Black women compared to white men. This misrepresentation reflects bias in training data and results in discriminatory treatment in surveillance or law enforcement scenarios.

 Harm Mitigation Strategies:
Inclusive and Balanced Training Data:
Ensure the dataset used to train AI systems includes a diverse representation of races, genders, ages, and other identity groups. This reduces the likelihood of skewed performance across demographic lines.

Bias Auditing and Algorithmic Impact Assessments:
Regularly audit models using fairness metrics and perform algorithmic impact assessments before deployment. These assessments help identify representational disparities and enforce accountability for their mitigation.

Legal and Ethical Implications of GenAI

5)Discuss the legal and ethical concerns of AI-generated content based on the examples of:
Memorizing private data (e.g., names in GPT-2)
Generating copyrighted material (e.g., Harry Potter text)
Do you believe generative AI models should be restricted from certain data during training? Justify your answer.

ANS)
1. Memorizing Private Data (e.g., names in GPT-2)
Concern:
Large language models like GPT-2 have shown they can memorize and regurgitate private or sensitive information from their training data — including names, phone numbers, or emails — even if those were never intended to be public.

Ethical issue:
This breaches user privacy and data protection laws such as the GDPR, which emphasizes the "right to be forgotten" and informed consent. If a model can leak personal data without consent, it undermines trust and violates ethical standards for privacy.

2. Generating Copyrighted Material (e.g., Harry Potter text)
Concern:
Generative models can reproduce copyrighted material they’ve been trained on — for example, generating text verbatim from books like Harry Potter. This raises intellectual property concerns, as AI may effectively "copy" protected works.

Legal issue:
This could violate copyright law, particularly if outputs are used commercially or distributed widely. The debate centers on whether training on copyrighted material without permission constitutes "fair use" or infringement.

 Should Generative AI Be Restricted from Certain Data During Training?
Yes — restrictions should apply.

Justification:
Privacy Protection:
Models trained on unrestricted web data are likely to ingest and retain sensitive or personal information. Consent, anonymization, and clear boundaries are essential to prevent harm.

Respect for Intellectual Property:
Training on copyrighted works without permission disregards the rights of creators. Restrictions — or licensing agreements — ensure that AI development respects existing legal frameworks and encourages ethical innovation.

6)Bias & Fairness Tools

Visit Aequitas Bias Audit Tool.
Choose a bias metric (e.g., false negative rate parity) and describe:
What the metric measures
Why it's important
How a model might fail this metric
Optional: Try applying the tool to any small dataset or use demo data.

ANS)
What the Metric Measures:
False Negative Rate (FNR) measures the proportion of actual positive cases that a model incorrectly classifies as negative (i.e., misses the positive case).
FNR Parity compares this rate across different groups (e.g., by race, gender) to detect disparities.

Formula:
FNR = False Negatives / (False Negatives + True Positives)
FNR Parity: The FNRs for different groups should be roughly equal.

 Why It's Important:
High false negative rates in sensitive domains can deny people critical services or protections. Ensuring parity prevents one group from being disproportionately overlooked.

Example:

In loan approvals, a false negative means denying a loan to someone who would have repaid it.

If the FNR is higher for one racial group, that group is unfairly disadvantaged — a form of allocational harm.

How a Model Might Fail This Metric:
Suppose a model is trained on biased historical data where certain groups were underrepresented in positive outcomes. It may:

Learn patterns that under-predict success for those groups.

Show higher FNRs for minorities, even when they qualify at the same rate as others.

Result: The model disproportionately misses eligible applicants from specific groups, failing FNR parity.








