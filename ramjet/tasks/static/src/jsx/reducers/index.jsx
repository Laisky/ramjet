import { createStore, combineReducers } from 'redux';
import { routerReducer as routing } from 'react-router-redux';

import logins from './login.jsx';


export const rootReducer = combineReducers({
    logins,
    routing
});

export const store = createStore(rootReducer);


