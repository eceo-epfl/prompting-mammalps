# Base image
FROM huggingface/transformers-pytorch-gpu

# Inpurt username, uid, gid
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GID
ARG LDAP_GROUPNAME

RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -U -s /bin/bash -G ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# Create home dir with same rights
RUN mkdir -p /home/${LDAP_USERNAME}
RUN chown ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}/
WORKDIR /home/${LDAP_USERNAME}
RUN pip install smolagents