import React, { createContext, useContext, useEffect, useState } from 'react';

const USERS_KEY = 'app_users_v1';
const CURRENT_KEY = 'app_current_user_v1';

// Seed a default admin on first load
const defaultUsers = {
  admin: { password: 'admin', role: 'admin' },
};

function readUsers() {
  try {
    const raw = localStorage.getItem(USERS_KEY);
    return raw ? JSON.parse(raw) : defaultUsers;
  } catch {
    return defaultUsers;
  }
}
function writeUsers(users) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [users, setUsers] = useState(() => readUsers());
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      const raw = localStorage.getItem(CURRENT_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });

  useEffect(() => writeUsers(users), [users]);

  useEffect(() => {
    if (currentUser)
      localStorage.setItem(CURRENT_KEY, JSON.stringify(currentUser));
    else localStorage.removeItem(CURRENT_KEY);
  }, [currentUser]);

  const login = (username, password) => {
    const u = users[username];
    if (!u || u.password !== password) throw new Error('Invalid credentials');
    setCurrentUser({ username, role: u.role });
  };

  const logout = () => setCurrentUser(null);

  const addUser = (username, password, role = 'user') => {
    if (!currentUser || currentUser.role !== 'admin')
      throw new Error('Only admin can add users');
    if (!username || !password)
      throw new Error('Username and password required');
    if (users[username]) throw new Error('User already exists');
    const next = { ...users, [username]: { password, role } };
    setUsers(next);
  };

  // Users can change their own password; admin can change theirs too.
  const changePassword = (username, oldPass, newPass) => {
    const target = users[username];
    if (!target) throw new Error('User not found');
    if (currentUser.username !== username && currentUser.role !== 'admin')
      throw new Error('Not allowed');

    // If changing your own password, verify old password
    if (currentUser.username === username && target.password !== oldPass)
      throw new Error('Old password incorrect');

    const next = { ...users, [username]: { ...target, password: newPass } };
    setUsers(next);
    if (currentUser.username === username)
      setCurrentUser({ username, role: target.role }); // keep session
  };

  const value = { users, currentUser, login, logout, addUser, changePassword };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);
