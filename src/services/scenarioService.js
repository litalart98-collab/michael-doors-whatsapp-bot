// ─── Scenario Classifier ──────────────────────────────────────────────────────
// Deterministic keyword-based classifier. Runs only on the FIRST message.
// When a scenario matches, the scripted response is returned and Claude is NOT
// called. All follow-up messages go directly to Claude.
//
// Detection order:
//   1.  greeting          — greeting only, no door content
//   2.  showroom_address  — address / location
//   3.  showroom_hours    — hours / appointment / visit
//   4.  repair            — repair / fault / past installation
//   5.  frame_removal     — פירוק משקוף question as opening
//   6.  designed_doors    — style/decorative keyword, no door type
//   7.  detailed_inquiry  — clear door type + purchase intent
//   8.  entrance_doors    — entrance door mention, browsing
//   9.  interior_doors    — interior door mention
//   10. vague_inquiry     — price/quote request, no door type
//   11. ambiguous         — generic door interest, no type

// ─── Helper predicates ────────────────────────────────────────────────────────
const hasEntrance     = m => /דלת כניסה|דלתות כניסה/.test(m);
const hasInterior     = m => /דלת פנים|דלתות פנים/.test(m);
const hasDoorType     = m => hasEntrance(m) || hasInterior(m);
const hasStyle        = m => /מודרנ|מעוצב|מעוצבת|קלאסי|קלאסית|חלקה|פשוטה/.test(m);
const isQuestion      = m => /\?|יש לכם|האם |אפשר /.test(m);
const hasFrameRemoval = m => /פירוק משקוף|עם פירוק|בלי פירוק|להוציא משקוף|להחליף משקוף|ללא פירוק/.test(m);

// Purchase / search intent — any natural phrasing of interest
const hasIntent = m =>
  /מתעניין|מתעניינת|מעוניין|מעוניינת|רוצה|צריך|צריכה|מחפש|מחפשת|מחפשים|מעניין אותי|מעניין אותנו|אשמח|מעוניינים/.test(m);

// Detailed = door type + intent + not just browsing/asking
const isDetailedEntrance = m => hasEntrance(m) && hasIntent(m) && !hasStyle(m) && !isQuestion(m);
const isDetailedInterior = m => hasInterior(m) && hasIntent(m) && !hasStyle(m) && !isQuestion(m);

// Greeting only — short message, no product content
const isGreetingOnly = m =>
  /^(שלום|היי|הי|בוקר טוב|ערב טוב|צהריים טובים|לילה טוב|אהלן|טוב|מה שלומכם|מה נשמע|ספריד|חחח|אוקי|אחלה|נהדר|מצוין)[.!,\s]*$/i.test(m.trim());

// ─── Scenario definitions ─────────────────────────────────────────────────────
const SCENARIOS = {

  greeting: {
    id: 'greeting',
    handoff_to_human: false,
    needs_frame_removal: null,
    summary: 'Greeting only — asking how to help',
    response: 'היי, תודה שפניתם לדלתות מיכאל, איך אפשר לעזור?',
  },

  showroom_address: {
    id: 'showroom_address',
    handoff_to_human: true,
    needs_frame_removal: null,
    summary: 'Customer asking for showroom address',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'הכתובת שלנו: בעלי המלאכה 15, נתיבות.\n' +
      'אם תרצו להגיע, אשמח שתשאירו שם מלא ומספר טלפון, וניצור איתכם קשר לתיאום.',
  },

  showroom_hours: {
    id: 'showroom_hours',
    handoff_to_human: true,
    needs_frame_removal: null,
    summary: 'Customer asking about hours or scheduling a showroom visit',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'נשמח לתאם עבורכם הגעה מסודרת לאולם התצוגה.\n' +
      'אשמח שתשאירו שם מלא, מספר טלפון ועיר מגורים, וניצור איתכם קשר לתיאום פגישה.',
  },

  repair: {
    id: 'repair',
    handoff_to_human: true,
    needs_frame_removal: null,
    summary: 'Customer requesting repair or service',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'אשמח שתכתבו שם מלא, עיר מגורים ומספר טלפון, וניצור איתכם קשר בהקדם.',
  },

  frame_removal: {
    id: 'frame_removal',
    handoff_to_human: false,
    needs_frame_removal: null,
    summary: 'Customer asking about frame removal — door type still unknown',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'אשמח להבין האם מדובר בדלת כניסה או דלת פנים, כדי שנוכל לכוון אתכם נכון.',
  },

  designed_doors: {
    id: 'designed_doors',
    handoff_to_human: false,
    needs_frame_removal: null,
    summary: 'Customer interested in designed/modern doors — door type not specified',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'כן, בהחלט, יש אצלנו מגוון דלתות מעוצבות ודגמים בסגנונות שונים.\n' +
      'אשמח להבין האם אתם מתעניינים בדלת כניסה או דלת פנים, כדי שנוכל לכוון אתכם נכון.',
  },

  detailed_inquiry: {
    id: 'detailed_inquiry',
    handoff_to_human: false,
    needs_frame_removal: null,
    summary: 'Customer specified entrance door with clear intent — asking about style next',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'האם אתם מחפשים דלת חלקה או מעוצבת?',
  },

  detailed_inquiry_interior: {
    id: 'detailed_inquiry_interior',
    handoff_to_human: false,
    needs_frame_removal: false,
    summary: 'Customer specified interior door with clear intent — asking for project context',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'אשמח שתכתבו אם מדובר בדלת פנים לבית, דירה או פרויקט, כדי שנוכל לכוון אתכם נכון.',
  },

  entrance_doors: {
    id: 'entrance_doors',
    handoff_to_human: false,
    needs_frame_removal: null,
    summary: 'Customer browsing entrance doors — guiding toward style preference',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'כן, בהחלט, יש אצלנו מגוון דלתות כניסה בסגנונות שונים, כולל דגמים מודרניים ודלתות מעוצבות.\n' +
      'אשמח שתכתבו אם אתם מחפשים דלת כניסה מודרנית, קלאסית או מעוצבת, ונכוון אתכם בהתאם.',
  },

  interior_doors: {
    id: 'interior_doors',
    handoff_to_human: false,
    needs_frame_removal: null,
    summary: 'Customer browsing interior doors — guiding toward project context',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'כן, בהחלט, יש אצלנו גם דלתות פנים במגוון סגנונות.\n' +
      'אשמח שתכתבו אם מדובר בדלת פנים לבית, דירה או פרויקט אחר, כדי שנוכל לכוון אתכם נכון.',
  },

  vague_inquiry: {
    id: 'vague_inquiry',
    handoff_to_human: false,
    needs_frame_removal: null,
    summary: 'Quote request without door type — asking for clarification',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'אשמח להבין באיזו דלת אתם מתעניינים — דלת כניסה או דלת פנים?',
  },

  ambiguous: {
    id: 'ambiguous',
    handoff_to_human: false,
    needs_frame_removal: null,
    summary: 'Vague door inquiry — asking entrance vs interior',
    response:
      'היי, תודה שפניתם לדלתות מיכאל.\n' +
      'אשמח להבין האם אתם מתעניינים בדלת כניסה או דלת פנים, כדי שנוכל לכוון אתכם נכון.\n' +
      'אחרי שתכתבו את הפרט הזה, נמשיך יחד בצורה מסודרת.',
  },

};

// ─── Main classifier ──────────────────────────────────────────────────────────
function detect(rawMessage) {
  const msg = rawMessage.trim();

  // 1. Greeting only
  if (isGreetingOnly(msg)) return SCENARIOS.greeting;

  // 2. Showroom address
  if (/כתובת|איפה אתם|איפה האולם|איפה החנות|המיקום שלכם|מיקום|להגיע אליכם|איך מגיעים/.test(msg)) {
    return SCENARIOS.showroom_address;
  }

  // 3. Showroom hours / appointment
  if (/שעות פעילות|שעות הפעילות|שעות פתיחה|שעות הפתיחה|מתי פתוח|מתי אפשר להגיע|לקבוע פגישה|תיאום פגישה|אולם תצוגה|אולם התצוגה/.test(msg)) {
    return SCENARIOS.showroom_hours;
  }

  // 4. Repair / service
  if (/תיקון|תקלה|התקנתם|הותקנה|שירות לדלת|בעיה בדלת|בעיה.*דלת|דלת.*בעיה|הדלת לא נסגרת|הדלת לא נפתחת|ציר שבור|ידית שבורה|אחריות/.test(msg)) {
    return SCENARIOS.repair;
  }

  // 5. Frame removal as opening — door type unknown
  if (hasFrameRemoval(msg) && !hasDoorType(msg)) return SCENARIOS.frame_removal;

  // 6. Designed/modern — style keyword, door type not yet specified
  if (hasStyle(msg) && !hasDoorType(msg)) return SCENARIOS.designed_doors;

  // 7. Detailed inquiry — door type + clear intent
  if (isDetailedEntrance(msg)) {
    // If style already mentioned — skip to frame removal question
    if (hasStyle(msg)) {
      return {
        ...SCENARIOS.detailed_inquiry,
        summary: 'Customer specified entrance door + style — asking about frame removal',
        response:
          'היי, תודה שפניתם לדלתות מיכאל.\n' +
          'האם יש צורך בפירוק משקוף קיים?',
      };
    }
    return SCENARIOS.detailed_inquiry;
  }
  if (isDetailedInterior(msg)) return SCENARIOS.detailed_inquiry_interior;

  // 8. Entrance doors — any mention
  if (hasEntrance(msg)) return SCENARIOS.entrance_doors;

  // 9. Interior doors — any mention
  if (hasInterior(msg)) return SCENARIOS.interior_doors;

  // 10. Vague quote request — no door type
  if (/הצעת מחיר|כמה עולה|כמה זה עולה|כמה עולים|מחיר|אפשר הצעה|מחיר.*דלת|דלת.*מחיר/.test(msg)) {
    return SCENARIOS.vague_inquiry;
  }

  // 11. Ambiguous door interest
  if (/מחפש דלת|מחפשת דלת|מחפשים דלת|מתעניין|מתעניינת|מתעניינים|מעוניין|מעוניינת|מעוניינים|דלת לבית|דלת לדירה|צריך דלת|צריכה דלת|אשמח למידע|אפשר פרטים|פרטים על|רוצה לדעת|ספרו לי|מה יש לכם|מה אתם מוכרים|מה אפשר|מה השירותים|מה המוצרים/.test(msg)) {
    return SCENARIOS.ambiguous;
  }

  return null; // falls through to Claude
}

module.exports = { detect };
