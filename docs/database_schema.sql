-- Planner database schema (PostgreSQL). GENERATED from service/db.py by scripts/export_schema.py; do not edit.
-- The service creates these tables itself on start-up (SQLite or PostgreSQL); this file documents them.
-- Transaction boundaries: see service/workflow.py (one transaction per state change, audit events included).

CREATE TABLE plans (
	id SERIAL NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	scenario JSON NOT NULL, 
	assignment JSON NOT NULL, 
	metrics JSON NOT NULL, 
	devices JSON NOT NULL, 
	activated_at TIMESTAMP WITH TIME ZONE, 
	superseded_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_plans_status CHECK (status IN ('candidate', 'active', 'superseded'))
);

CREATE UNIQUE INDEX uq_plans_one_active ON plans (status) WHERE status = 'active';

CREATE TABLE proposals (
	id SERIAL NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	idempotency_key VARCHAR(128), 
	plan_id INTEGER NOT NULL, 
	base_plan_id INTEGER, 
	status VARCHAR(24) NOT NULL, 
	reason VARCHAR(16) NOT NULL, 
	objective_current FLOAT, 
	objective_new FLOAT NOT NULL, 
	util_new FLOAT NOT NULL, 
	migration JSON NOT NULL, 
	checks JSON NOT NULL, 
	needs_human BOOLEAN NOT NULL, 
	applied_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_proposals_status CHECK (status IN ('rejected', 'awaiting_approval', 'ready', 'applied', 'denied', 'stale')), 
	CONSTRAINT ck_proposals_reason CHECK (reason IN ('initial', 'rebalance', 'restore')), 
	UNIQUE (idempotency_key), 
	FOREIGN KEY(plan_id) REFERENCES plans (id), 
	FOREIGN KEY(base_plan_id) REFERENCES plans (id)
);

CREATE INDEX ix_proposals_status ON proposals (status);

CREATE TABLE approvals (
	id SERIAL NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	proposal_id INTEGER NOT NULL, 
	approver VARCHAR(128) NOT NULL, 
	decision VARCHAR(8) NOT NULL, 
	comment TEXT NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_approvals_decision CHECK (decision IN ('approve', 'deny')), 
	UNIQUE (proposal_id), 
	FOREIGN KEY(proposal_id) REFERENCES proposals (id)
);

CREATE TABLE audit_events (
	id SERIAL NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	actor VARCHAR(160) NOT NULL, 
	event VARCHAR(40) NOT NULL, 
	proposal_id INTEGER, 
	plan_id INTEGER, 
	detail JSON NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(proposal_id) REFERENCES proposals (id), 
	FOREIGN KEY(plan_id) REFERENCES plans (id)
);

CREATE INDEX ix_audit_events_proposal ON audit_events (proposal_id);
