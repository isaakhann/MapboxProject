import React, { useState } from 'react';
import { useAuth } from '../auth/AuthContext';

export default function SettingsModal({ open, onClose }) {
  const { currentUser, addUser, changePassword, logout } = useAuth();
  const isAdmin = currentUser?.role === 'admin';

  // Change own password
  const [oldPass, setOldPass] = useState('');
  const [newPass, setNewPass] = useState('');
  const [pwMsg, setPwMsg] = useState('');

  // Add user (admin only)
  const [newUser, setNewUser] = useState('');
  const [newUserPass, setNewUserPass] = useState('');
  const [newRole, setNewRole] = useState('user');
  const [addMsg, setAddMsg] = useState('');

  if (!open) return null;

  const submitChange = (e) => {
    e.preventDefault();
    setPwMsg('');
    try {
      changePassword(currentUser.username, oldPass, newPass);
      setOldPass('');
      setNewPass('');
      setPwMsg('Password changed.');
    } catch (e) {
      setPwMsg(e.message || 'Failed to change password.');
    }
  };

  const submitAdd = (e) => {
    e.preventDefault();
    setAddMsg('');
    try {
      addUser(newUser.trim(), newUserPass, newRole);
      setNewUser('');
      setNewUserPass('');
      setNewRole('user');
      setAddMsg('User added.');
    } catch (e) {
      setAddMsg(e.message || 'Failed to add user.');
    }
  };

  return (
    <div className="modal" style={{ display: 'flex' }}>
      <div className="modal-content" style={{ width: 520 }}>
        <div className="modal-header">
          <span>Settings</span>
          <span className="close" onClick={onClose}>
            &times;
          </span>
        </div>

        <div className="modal-body">
          <h3 style={{ marginTop: 0 }}>Account</h3>
          <form onSubmit={submitChange} className="settings-form">
            <label>Old password</label>
            <input
              type="password"
              value={oldPass}
              onChange={(e) => setOldPass(e.target.value)}
              required
            />
            <label>New password</label>
            <input
              type="password"
              value={newPass}
              onChange={(e) => setNewPass(e.target.value)}
              required
            />
            {pwMsg && <div className="auth-msg">{pwMsg}</div>}
            <button className="auth-btn" type="submit">
              Change password
            </button>
          </form>

          {isAdmin && (
            <>
              <hr style={{ margin: '18px 0' }} />
              <h3>Add user</h3>
              <form onSubmit={submitAdd} className="settings-form">
                <label>Username</label>
                <input
                  value={newUser}
                  onChange={(e) => setNewUser(e.target.value)}
                  required
                />
                <label>Password</label>
                <input
                  type="password"
                  value={newUserPass}
                  onChange={(e) => setNewUserPass(e.target.value)}
                  required
                />
                <label>Role</label>
                <select
                  value={newRole}
                  onChange={(e) => setNewRole(e.target.value)}
                >
                  <option value="user">User</option>
                  <option value="admin">Admin</option>
                </select>
                {addMsg && <div className="auth-msg">{addMsg}</div>}
                <button className="auth-btn" type="submit">
                  Add user
                </button>
              </form>
            </>
          )}

          <hr style={{ margin: '18px 0' }} />
          <button className="btn-logout" onClick={logout}>
            Log out
          </button>
        </div>
      </div>
    </div>
  );
}
