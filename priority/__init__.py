"""Priority score P = 100*clip(wS*S + wE*E + wV*V + wT*T + wM*M) (PLAN section 6; WP5).

Pure library (no DB / server imports). Entry point: ``priority.score.compute(pothole, ctx)``.
"""
