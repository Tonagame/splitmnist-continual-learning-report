תיאור ה-flow המרכזי של כל העבודה עם AI:  
התחלנו מהתקנה והרצת קוד קיים, עברנו לשחזור תוצאות, אחר כך למימוש עצמאי, אחר כך לניסוי H&T, ואז לארגון GitHub, גרפים, מסמכים והכנה להגנה.
## 1. הקמת הסביבה והרצת הקוד המקורי
1. להריץ את GMvandeVen continual-learning repository מקומית על Windows עם RTX 3070.
2. לבדוק Python, Conda, Git, NVIDIA driver ו-PyTorch CUDA.
3. ליצור Conda environment בשם `continual` עם Python 3.10.
4. להתקין PyTorch עם CUDA.
5. לשכפל את הריפו.
6. להתקין dependencies.
7. לוודא ש-`torch.cuda.is_available()` מחזיר True ושה-GPU הוא RTX 3070.
8. לקרוא את README ולמצוא פקודת MNIST קטנה.
9. להריץ smoke test.
10. לתקן שגיאות אם מופיעות.
11. להעביר את כל העבודה מ-`C:` ל-`E:`.
12. לבדוק האם MNIST כבר קיים ב-dataset.

## 2. הרצת Baselines ראשונים
1. להריץ `None` ו-`Joint` על Split MNIST.
2. להריץ עבור שלושת התרחישים:
 - Task-CL
 - Domain-CL
 - Class-CL
3. ליצור גרף שמשווה None מול Joint.
4. לדווח:
 - הפקודות המדויקות
 - accuracy
 - האם משהו נכשל
 - איפה נשמרו התוצאות
5. לשאול האם זה על כל MNIST.

## 3. שיטות קלאסיות
1. להריץ:
 - EWC
 - LwF
 - A-GEM
 - Generative Classifier אם נתמך
 - Separate Networks רק ל-Task-CL
2. ליצור:
 - `summary.csv`
 - גרף השוואה
 - logs לכל ריצה
3. לדווח:
 - פקודות מדויקות
 - final accuracy
 - מי הכי קרוב ל-Joint
 - מי הכי שיפר מעל None
 - האם משהו נכשל
4. להריץ את אותן שיטות על Task-CL ו-Domain-CL.
5. להריץ את אותן שיטות על Class-CL.
6. לוודא Class-CL אמיתי:
 - בלי task identity
 - בלי allowed classes
 - הערכה על כל 10 המחלקות.

## 4. גרפים וסיכומים
1. ליצור גרף שמשווה כל טכניקה ב:
 - Task
 - Domain
 - Class
2. ליצור גרפים לפי scenario.
3. ליצור גרפים סופיים של accuracies.
4. להציג איפה הגרפים נשמרו.
5. ליצור summary מלא של מה למדנו.
6. ליצור Word בעברית עם כל הגרפים.
7. לתקן בעיית encoding שבה Word יצא עם `????`.

## 5. שיטת LSR-lite / H&T
1. לבדוק שני prototypes:
 - LSR-lite
 - LSR-lite + Fourier
2. להגדיר LSR-lite:
 - replay buffer עם דוגמאות train אמיתיות
 - שמירת image ו-label
 - שמירת teacher logits
 - שמירת penultimate feature vector
 - loss של CE + KD + feature anchoring
3. להוסיף Fourier רק כ-auxiliary loss, לא כתחליף ל-replay.
4. לוודא:
 - אין test data באימון
 - ה-buffer נבנה רק מ-train
 - לא שומרים את כל הדאטה הישן
 - אותו buffer budget כמו A-GEM
 - אותם settings
5. לשאול האם ל-LSR יש Adaptive Stability Weighting.
6. להוסיף:
 - LSR-lite + ASW
 - LSR-lite + Fourier + ASW
7. להגדיר ASW:
 - `adaptive_factor = old_loss / (new_loss + epsilon)`
 - clamp בין 0.5 ל-2.0
 - התאמה דינמית של `lambda_kd` ו-`lambda_feat`
8. להריץ קודם smoke test עם `iters=1`.
9. אם עובד, להריץ `iters=100`.
10. ליצור graph.
11. להסביר איך LSR יושם.
12. לשנות את השם מ-LSR ל-H&T.
13. להסביר ש-H&T הוא hybrid בין:
 - replay/memory בסגנון A-GEM
 - distillation בסגנון LwF
 - feature anchoring
14. להבהיר שהוא לא באמת EWC כי אין Fisher penalty.

## 6. ניסויי 2000 רציניים
1. Phase 1: להריץ Split MNIST Class-CL עם:
 - contexts=5
 - iters=2000
 - batch=128
 - acc-n=1024
 - RTX 3070
2. להריץ:
 - None
 - EWC
 - LwF
 - A-GEM
 - Generative Classifier
 - H&T
 - H&T + Fourier
 - H&T + ASW
 - H&T + Fourier + ASW
 - Joint
3. לשמור לתיקייה חדשה ולא לדרוס תוצאות 100 iterations.
4. להעריך כל 10 iterations.
5. ליצור:
 - `summary.csv`
 - logs
 - final accuracy graph
 - learning curve graph
 - `learning_curve.csv`
 - report
6. Phase 2: להריץ Domain-CL 2000, בלי Task-CL.
7. Phase 3: להריץ Task-CL 2000, בלי Domain-CL.
8. לדווח runtime, gaps, improvements, failures.
9. לשאול למה Generative Classifier נכשל.
10. לשאול למה Generative Classifier נמוך מהמאמר.
11. להסיר Generative Classifier לגמרי מהגרפים ומהתיעוד.

## 7. מימוש עצמאי
1. לשאול האם מימשנו את השיטות מהמאמר או רק הרצנו קוד מוכן.
2. להבהיר שאסור לקחת את הקוד מה-GitHub המקורי.
3. לבנות מימוש עצמאי שמשחזר את התוצאות.
4. להריץ את כל השיטות הקלאסיות מהמימוש שלנו.
5. להשוות:
 - תוצאות המאמר
 - הרצת הקוד המקורי
 - המימוש שלנו
6. ליצור גרף השוואה בין שלושת המקורות.
7. לבדוק למה A-GEM שלנו שונה מהמאמר.
8. לבדוק למה EWC רחוק מהמאמר.
9. לבדוק האם הרצות קודמות לא היו באמת CL כי נשמרו פרמטרים ישנים.
10. לבדוק האם המימוש עומד בדרישות.

## 8. יצירת עמוד GitHub ותיעוד
1. ליצור GitHub repository.
2. לכתוב description.
3. להגדיר configuration.
4. להעלות את הקוד וההסברים ל-GitHub.
5. ליצור GitHub page יפה עם:
 - machine learning explanations
 - איך המודל עובד
 - איך השיטות עובדות
 - גרפים
 - תוצאות
6. לסדר את GitHub כי הוא נהיה messy.
7. להוסיף comments/docstrings לכל הפונקציות.
8. לפצל את הקוד כך שלכל method יהיה קובץ משלו.
9. להוסיף README ל-assets שמציג את כל הגרפים.
10. למחוק קבצים לא רלוונטיים.
11. להסיר את הביטוי `clean room` מכל GitHub.
12. להסיר `docs2` ולשנות ל-`doc`.
13. ליצור `doc` עם:
 - algorithmic thinking
 - AI documentation
 - takeaways
 - results and comparison
14. ליצור README פשוט לתיקיית doc.
15. להוסיף takeaway אישי של Haim.
16. לעדכן את README הראשי:
 - גרף ראשון: הרצת הקוד שלהם מול המאמר
 - גרף שני: הקוד שלנו מול המאמר והקוד שלהם
 - גרף שלישי: H&T מול שאר השיטות
17. להסיר submission checklist ו-GitHub page setup מה-README.

## 9. שאלות והסברים להבנה
1. למה ב-Domain-CL `output_dim = 2`, וב-Class/Task `output_dim = 10`?
2. איך Task, Domain ו-Class עובדים?
3. האם `evaluate_neural` הוא החיזוי?
4. איפה המודל חוזה?
5. למה יש שיטה שלא משתמשת ב-backpropagation?
6. איך ב-EWC מונעים מהעונש להיות גדול מדי?
7. כמה iterations עשינו?
8. האם Fourier loss דומה ל-gradient conflict ב-A-GEM?
9. האם Fourier הוא component מרכזי או guardrail?
10. להסביר יותר על `add_lsr_replay_loss`.
11. להסביר מה זה:
 - correct label
 - old model output behavior
 - old internal representation
12. האם H&T לוקח הרבה זיכרון?
13. האם H&T מתאים לדאטהסטים גדולים?
14. האם הוא שומר את כל הדאטה?
15. איך להפוך אותו ל-memory efficient?
16. האם H&T הוא hybrid של A-GEM, LwF ו-EWC?
17. מה זה feature anchoring?
18. A-GEM מחשב gradient של מה?


