from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointsV2(object):
    """All available Hubble API endpoints."""

    initiate_session_authorize: str = 'user.identity.authorize'
    initiate_proxied_authorize: str = 'user.identity.proxiedAuthorize'
    auto_grant_user_identity: str = 'user.identity.grant.auto'
    get_user_info: str = 'user.identity.whoami'
    dismiss_user_session: str = 'user.session.dismiss'

    create_pat: str = 'user.pat.create'
    list_pats: str = 'user.pat.list'
    delete_pat: str = 'user.pat.delete'

    upload_artifact: str = 'artifact.upload'
    download_artifact: str = 'artifact.getDownloadUrl'
    delete_artifact: str = 'artifact.delete'
    delete_multiple_artifacts: str = 'artifact.deleteMany'
    get_artifact_info: str = 'artifact.getDetail'
    list_artifacts: str = 'artifact.list'
    update_artifact: str = 'artifact.update'

    list_internal_docker_registries: str = 'dockerRegistry.listInternalRegistries'
