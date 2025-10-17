import React, { useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import loginImg from '../img/login.png';

export default function Login() {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [remember, setRemember] = useState(true);
  const [err, setErr] = useState('');

  const onSubmit = (e) => {
    e.preventDefault();
    setErr('');
    try {
      login(username.trim(), password);
    } catch (e) {
      setErr(e.message || 'Login failed');
    }
  };

  return (
    <div className="login-shell">
      <div className="login-grid">
        <form className="login-card" onSubmit={onSubmit}>
          <div className="brand-row">
            <div className="brand-badge" />
            <div>
              <h2 className="login-title">Welcome!</h2>
              <p className="login-sub">Sign in to your account.</p>
            </div>
          </div>

          <label className="field-label">Username*</label>
          <div className="field">
            <span className="icon">
              <svg width="18" height="18" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M12 12a5 5 0 1 0-5-5a5 5 0 0 0 5 5Zm0 2c-5.33 0-8 2.67-8 6v1h16v-1c0-3.33-2.67-6-8-6Z"
                />
              </svg>
            </span>
            <input
              className="input"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoFocus
            />
          </div>

          <label className="field-label">Password*</label>
          <div className="field">
            <span className="icon">
              <svg width="18" height="18" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M17 8h-1V6a4 4 0 1 0-8 0v2H7a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2Zm-6 7.73V17h2v-1.27a2 2 0 1 0-2 0ZM9 6a3 3 0 1 1 6 0v2H9Z"
                />
              </svg>
            </span>
            <input
              className="input"
              placeholder="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <div className="login-row">
            <label className="remember">
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
              />
              Remember me
            </label>
          </div>

          {err && <div className="auth-error">{err}</div>}

          <button className="btn-primary" type="submit">
            Login
          </button>
          <p className="auth-hint"></p>
        </form>

        <div className="hero-panel">
          <img className="hero-img" src={loginImg} alt="App preview" />
          <div className="hero-vignette" />
        </div>
      </div>
    </div>
  );
}
