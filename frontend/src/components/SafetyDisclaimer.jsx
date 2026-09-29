import React from 'react';

export default function SafetyDisclaimer() {
  return (
    <section className="card safety-card" role="note">
      <span className="safety-icon" aria-hidden="true">🛡️</span>
      <div>
        <strong>Educational information, not medical advice</strong>
        <p>
          MediQ can explain symptoms and reports in plain language, but it cannot diagnose you and
          is not a substitute for a clinician. If you have emergency symptoms — such as chest pain,
          trouble breathing, or sudden weakness — contact your local emergency services now.
        </p>
      </div>
    </section>
  );
}
