export function SiteHeader({ research = false }: { research?: boolean }) {
  return (
    <header className="site-header">
      <div className="brand" aria-label="FCKFACE">
        <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
        <span>FCKFACE</span>
      </div>
      <span className="header-tag">{research ? 'LOCAL RESEARCH PREVIEW' : 'OPEN RESEARCH PROJECT'}</span>
    </header>
  )
}

export function ResearchStatus() {
  return (
    <main className="page status-page">
      <div className="eyebrow"><span className="signal" /> RESEARCH IN PROGRESS / 001</div>
      <h1>Can an image effect make face matching less reliable?</h1>
      <p className="lede">We are developing and testing visible photo alterations against multiple facial recognition models and common image processing paths.</p>
      <div className="status-rule" />
      <div className="status-grid">
        <div><span className="overline">CURRENT STATUS</span><h2>Testing the method</h2><p>No photo processor is available here yet. A public tool will follow only if an approach meets our independent evaluation standard.</p></div>
        <div><span className="overline">THE STANDARD</span><h2>Evidence before release</h2><p>A visual effect alone does not establish facial privacy. Read the <a href="https://github.com/iharc-jordan/FCKFACE/blob/main/research/results.md">development findings</a>, including failed approaches and current limitations.</p></div>
      </div>
      <footer className="page-footer"><span>FCKFACE / Face Cloaking Kit for Feature Alteration and Cross-model Evaluation</span><span><a href="https://github.com/iharc-jordan/FCKFACE">Source code</a> · No photos collected</span></footer>
    </main>
  )
}
